# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
import random
from functools import partial
from io import BytesIO
from pathlib import Path

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

from dinov2.train.ssl_meta_arch import SSLMetaArch
from datasets import IterableDatasetDict, load_dataset, DownloadConfig
from PIL import Image
import torch.utils.data


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")
import wandb


def _build_streaming_dataset(
    dataset_path: str,
    *,
    shuffle_buffer: int,
    shuffle_seed: int,
    world_size: int,
    global_rank: int,
    fragment_prefetch_limit: int,
    fragment_range_size: int,
):
    import pyarrow
    import pyarrow.dataset

    fragment_scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
        cache_options=pyarrow.CacheOptions(
            prefetch_limit=fragment_prefetch_limit,
            range_size_limit=fragment_range_size,
        ),
    )
    dataset = load_dataset(
        dataset_path,
        streaming=True,
        fragment_scan_options=fragment_scan_options,
    )["train"]
    dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=shuffle_seed)
    if world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=global_rank)
    return dataset

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)
    
    from omegaconf import OmegaConf
    if distributed.is_main_process():
        run = wandb.init(
            project="midnight-rep",  # Specify your project
            config = OmegaConf.to_container(cfg)
        )



    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

    #This is where we resume. This error is veyr strange thuogh...
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    """W20250825 03:00:27 4071999 fvcore.common.checkpoint checkpoint.py:352] The checkpoint state_dict contains keys that are not used by the model:
  student.backbone._flat_param
  student.backbone.blocks.0._flat_param
  student.backbone.blocks.1._flat_param
  student.backbone.blocks.2._flat_param
  student.backbone.blocks.3._flat_param
  student.dino_head._flat_param
  student.ibot_head._flat_param
  teacher.backbone._flat_param
  teacher.backbone.blocks.0._flat_param
  teacher.backbone.blocks.1._flat_param
  teacher.backbone.blocks.2._flat_param
  teacher.backbone.blocks.3._flat_param
  teacher.dino_head._flat_param
  teacher.ibot_head._flat_param
    """

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    early_stop_iter = cfg.optim.early_stop * OFFICIAL_EPOCH_LENGTH
    eta_target_iter = min(max_iter, early_stop_iter)

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loader

    # print("dataset path is", cfg.train.dataset_path)#This is the imagenet string shit
    # dataset = make_dataset(
    #     dataset_str=cfg.train.dataset_path,
    #     transform=data_transform,
    #     target_transform=lambda _: (),
    # )
    # # sampler_type = SamplerType.INFINITE
    # sampler_type = SamplerType.SHARDED_INFINITE
    # data_loader = make_data_loader(
    #     dataset=dataset,
    #     batch_size=cfg.train.batch_size_per_gpu,
    #     num_workers=cfg.train.num_workers,
    #     #num_workers = 1,
    #     shuffle=True,
    #     seed=0,
    #     sampler_type=sampler_type,
    #     sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
    #     drop_last=True,
    #     collate_fn=collate_fn,
    # )

    # import litdata as ld
    
    # dataset_root = "/data/TCGA_lit_sample30"
    
    # def decode_and_transform(item):
    #     image_file = BytesIO(item["image_bytes"])
    #     image = Image.open(image_file)
    #     image = image.convert("RGB")
    #     transformed = data_transform(image)
    #     slide_meta = (item["slide_path"], item["x"], item["y"], item["level"])
    #     return (transformed, None), slide_meta
    
    # dataset = ld.StreamingDataset(
    #     input_dir=dataset_root,
    #     shuffle=True,
    #     drop_last=True,
    #     seed=0,
    #     transform=decode_and_transform,
    #     max_cache_size="1000GB",
    # )
    
    # data_loader = ld.StreamingDataLoader(
    #     dataset,
    #     batch_size=cfg.train.batch_size_per_gpu,
    #     num_workers=cfg.train.num_workers,
    #     drop_last=True,
    #     pin_memory=True,
    #     collate_fn=collate_fn,
    # )

    dataset_builder = partial(
        _build_streaming_dataset,
        dataset_path="medarc/TCGA-12K-parquet",
        shuffle_buffer=10000,
        shuffle_seed=42,
        world_size=distributed.get_global_size(),
        global_rank=distributed.get_global_rank(),
        fragment_prefetch_limit=1,
        fragment_range_size=128 << 20,
    )

    def decode_and_transform(item):
        image_file = BytesIO(item["image_bytes"])
        image = Image.open(image_file)
        image = image.convert("RGB")
        transformed = data_transform(image)
        slide_meta = (item["slide_path"], item["x"], item["y"], item["level"])
        return (transformed, None), slide_meta

    class _TransformedStreamingDataset(torch.utils.data.IterableDataset):
        def __init__(self, dataset_builder, transform):
            self._dataset_builder = dataset_builder
            self._transform = transform

        def __iter__(self):
            # Build the HF streaming dataset inside the worker process to keep file handles valid.
            source = self._dataset_builder()
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None and worker_info.num_workers > 1:
                source = source.shard(num_shards=worker_info.num_workers, index=worker_info.id)
            for sample in source:
                yield self._transform(sample)

    dataset = _TransformedStreamingDataset(dataset_builder, decode_and_transform)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    import time
    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        eta_target_iter,
        start_iter,
    ):
        if iteration >= early_stop_iter:
            logger.info("Early stopping at iteration {}".format(iteration))
            if cfg.evaluation.eval_period_iterations >= 0:
                do_test(cfg, model, f"training_{iteration}")
                torch.cuda.synchronize()
            checkpointer.save(f"model_{iteration:07d}", iteration=iteration)
            break
        start = time.time() 
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return
        
        
        nan_mask = torch.isnan(data["collated_global_crops"])
        nan_mask2 = torch.isnan(data["collated_local_crops"])
        if nan_mask.any():
            print("found nan in input data")
            print(data[indexes])
        

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses

        optimizer.zero_grad(set_to_none=True)

        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            print(sum(loss_dict_reduced.values()))
            logger.info("NaN detected")
            print(data["indexes"])
            
            for name, param in model.named_parameters():
                if torch.isnan(param.data).any():
                    print(f"NaNs found in parameter: {name}")

            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)
        
        if distributed.is_main_process():
            wandb.log({"Learning Rate":lr,
                        "Momentum": mom,
                        "Last Layer LR": last_layer_lr,
                        "Learning Rate":lr,
                        "Total Loss":losses_reduced
                })
            wandb.log(loss_dict)


        
        # checkpointing and testing
        
        end_event.record()
    
        # Synchronize the GPU to ensure all operations are complete before measuring
        torch.cuda.synchronize()
        # Calculate and print the elapsed time
        elapsed_ms = start_event.elapsed_time(end_event)
        # print(f"Time elapsed for matmul: {elapsed_ms:.2f} ms")
        
        #Save instantly
        if cfg.evaluation.eval_period_iterations >= 0 and (iteration) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
        end = time.time()
        # print(end - start)
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)
    print(cfg)
    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    #Load model here from pretrained.
    if cfg.train.use_pretrained:

        if cfg.student.arch == "vit_giant2":
            print("loading pretraind DinoV2-giant") 
            model_pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
        else:
            AssertionError("only giant pretrained supported currently")

        model_pretrained = model_pretrained.to(torch.device("cuda"))
        model.student.backbone.patch_embed.proj.weight = model_pretrained.patch_embed.proj.weight
        model.student.backbone.patch_embed.proj.bias = model_pretrained.patch_embed.proj.bias
        model.student.backbone.cls_token = model_pretrained.cls_token
        model.student.backbone.register_tokens = model_pretrained.register_tokens
        model.student.backbone.mask_token = model_pretrained.mask_token

        print(model.state_dict().keys())
        print(model_pretrained.state_dict().keys())
        print(model_pretrained.pos_embed.shape) #1, 1360, 384. We lose pos embed because it was 518
        print(model.student.backbone.pos_embed.shape) #1, 257, 384

        #Interpolate pos embed
        if True: # small only
            source = model_pretrained.pos_embed.transpose(1, 2)
            size = model.student.backbone.pos_embed.shape[1]
            interpolated = torch.nn.functional.interpolate(source, size=size, mode='linear', align_corners=False)
            interpolated_embeddings = interpolated.transpose(1, 2)
            model.student.pos_embed = interpolated_embeddings
        else:
            model.student.pos_embed = model_pretrained.pos_embed

        # We need to make sure we grab *all* of the keys.
        # For each block, copy weights over.
        layers = []
        for layer in model_pretrained.blocks:
            layers.append(layer)
        i = 0
        for layer in model.student.backbone.blocks:
            for sublayer in layer:
                if type(sublayer) != torch.nn.Identity:
                    #So we have the subblock, now we need to convert
                    current = layers.pop(0)
                    sublayer.norm1.weight = current.norm1.weight
                    sublayer.norm1.bias = current.norm1.bias
                    
                    sublayer.attn.qkv.weight = current.attn.qkv.weight
                    #
                    sublayer.attn.qkv.bias =  current.attn.qkv.bias

                    sublayer.attn.proj.weight = current.attn.proj.weight
                    
                    #
                    sublayer.attn.proj.bias = current.attn.proj.bias


                    sublayer.norm2.weight = current.norm2.weight
                    
                    #
                    sublayer.norm2.bias = current.norm2.bias
                    try:
                        sublayer.mlp.fc1.weight = current.mlp.fc1.weight
                        sublayer.mlp.fc2.weight = current.mlp.fc2.weight
                        sublayer.mlp.fc1.bias = current.mlp.fc1.bias
                        sublayer.mlp.fc2.bias = current.mlp.fc2.bias
                    except:#Not very clean, needs improvement
                        sublayer.mlp.w12.weight = current.mlp.w12.weight
                        sublayer.mlp.w12.bias = current.mlp.w12.bias

                        sublayer.mlp.w3.weight = current.mlp.w3.weight
                        sublayer.mlp.w3.bias = current.mlp.w3.bias
                        


                    sublayer.ls1.gamma = current.ls1.gamma
                    sublayer.ls2.gamma = current.ls2.gamma


        model.student.backbone.norm.weight = model_pretrained.norm.weight
        
        #
        model.student.backbone.norm.bias = model_pretrained.norm.bias 

        #The temp sucess was norm bias off.

    #freeze everything *Except* position embeddings and new params
    model.prepare_for_distributed_training()
    if False:#Test partial freeze
        for param in model.student.parameters():
            param.requires_grad = False
        for param in model.student.ibot_head.parameters():
            param.requires_grad = True
        for param in model.student.dino_head.parameters():
            param.requires_grad = True


        if False:#For giant, we don't need to tune this
            model.student.backbone.pos_embed.requires_grad = True

    logger.info("Model:\n{}".format(model))

    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
