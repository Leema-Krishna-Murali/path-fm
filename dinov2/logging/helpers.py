# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from collections import defaultdict, deque
import datetime
import json
import logging
import time
from typing import Optional, Iterator, Any

import torch

import dinov2.distributed as distributed


logger = logging.getLogger("dinov2")


class MetricLogger(object):
    def __init__(self, delimiter="\t", output_file=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_file = output_file

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None or not distributed.is_main_process():
            return
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump.update({k: v.median for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")

    def log_every(self, iterable, print_freq, header=None, n_iterations=None, start_iteration=0):
        """Optimized version for StreamingDataset that avoids iterator recreation"""
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")

        # Try to get length efficiently
        if n_iterations is None:
            try:
                n_iterations = len(iterable)
            except (TypeError, AttributeError):
                # If len() not available, must count
                logger.warning("Cannot determine dataset length, progress reporting may be inaccurate")
                n_iterations = float('inf')

        space_fmt = ":" + str(len(str(n_iterations))) + "d" if n_iterations != float('inf') else ""

        # Pre-build log message template
        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]" if n_iterations != float('inf') else "[{0}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            log_list += ["max mem: {memory:.0f}"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0

        # Check if we should log on this rank
        should_log = distributed.is_main_process() if distributed.is_enabled() else True
        
        # Pre-compute print iterations for faster checking
        print_iterations = set(range(0, n_iterations if n_iterations != float('inf') else 1000000, print_freq))
        if n_iterations != float('inf'):
            print_iterations.add(n_iterations - 1)

        # Single iteration through the dataset
        for batch_idx, obj in enumerate(iterable, start=i):
            # Measure data loading time
            data_time.update(time.time() - end)
            
            # Yield the batch
            yield obj
            
            # Clear reference immediately
            obj = None
            
            # Measure full iteration time
            iter_time.update(time.time() - end)
            
            # Log if needed (optimized check)
            if should_log and (batch_idx in print_iterations or (n_iterations == float('inf') and batch_idx % print_freq == 0)):
                self.dump_in_output_file(iteration=batch_idx, iter_time=iter_time.avg, data_time=data_time.avg)
                
                if n_iterations != float('inf'):
                    eta_seconds = iter_time.global_avg * (n_iterations - batch_idx)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                else:
                    eta_string = "unknown"
                
                if has_cuda:
                    logger.info(
                        log_msg.format(
                            batch_idx,
                            n_iterations if n_iterations != float('inf') else "?",
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            batch_idx,
                            n_iterations if n_iterations != float('inf') else "?",
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            
            end = time.time()
            
            # Early exit if we know the length
            if n_iterations != float('inf') and batch_idx >= n_iterations - 1:
                break

        # Final statistics
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        if n_iterations != float('inf'):
            actual_iterations = min(batch_idx + 1, n_iterations)
            logger.info("{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / actual_iterations))
        else:
            logger.info("{} Total time: {}".format(header, total_time_str))

    def log_every_streaming(self, iterable, print_freq, header=None, max_iterations=None, start_iteration=0):
        """Alternative method optimized specifically for streaming/infinite datasets"""
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")

        log_list = [
            header,
            "[{0}]",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            log_list += ["max mem: {memory:.0f}"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0

        should_log = distributed.is_main_process() if distributed.is_enabled() else True
        
        for batch_idx, obj in enumerate(iterable, start=i):
            # Check max iterations
            if max_iterations is not None and batch_idx >= max_iterations:
                break
                
            # Measure data loading time
            data_time.update(time.time() - end)
            
            # Yield the batch
            yield obj
            
            # Clear reference
            obj = None
            
            # Measure full iteration time  
            iter_time.update(time.time() - end)
            
            # Log periodically
            if should_log and batch_idx % print_freq == 0:
                self.dump_in_output_file(iteration=batch_idx, iter_time=iter_time.avg, data_time=data_time.avg)
                
                if has_cuda:
                    logger.info(
                        log_msg.format(
                            batch_idx,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            batch_idx,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            
            end = time.time()

        # Final statistics
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("{} Total time: {}".format(header, total_time_str))


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, num=1):
        self.deque.append(value)
        self.count += num
        self.total += value * num

    def synchronize_between_processes(self):
        """
        Distributed synchronization of the metric
        Warning: does not synchronize the deque!
        """
        if not distributed.is_enabled():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )