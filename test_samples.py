

with open("sample_dataset.py", "r") as f:
    for line in f.readlines():
        print(line)
        path, x, y, level = line.split(" ")
