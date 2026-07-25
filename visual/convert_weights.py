import struct
import os
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
weights_path = os.path.join(repo_root, 'training', 'weights_best.pth')
output_path = os.path.join(script_dir, 'core', 'assets', 'weights.fw')

sd = torch.load(weights_path, map_location='cpu', weights_only=True)
keys = list(sd.keys())

print(f"Concatenating {len(keys)} layers:")
total = 0
arrays = []
for key in keys:
    arr = sd[key].numpy().astype('float32').flatten()
    arrays.append(arr)
    total += len(arr)
    print(f"  {key}: {sd[key].shape} -> {len(arr)} floats  (offset before: {total - len(arr)})")

all_weights = np.concatenate(arrays)
all_weights.tofile(output_path)

file_size = os.path.getsize(output_path)
print(f"Wrote {output_path} ({file_size} bytes, {total} floats, expected {total * 4})")
assert file_size == total * 4
assert file_size == 1268336
