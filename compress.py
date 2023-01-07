import os
import argparse
import torch

def compress(checkpoint_path, output_path):
    loaded_obj = torch.load(checkpoint_path, map_location='cpu')
    loaded_obj.pop('optimizer')
    
    torch.save(loaded_obj, output_path)


if __name__ == "__main__":
    base_dir = '/debug/compress/'
    for f in os.listdir(base_dir):
        p = os.path.join(base_dir, f)
        if os.path.isfile(p):
            compress(p, os.path.join(base_dir, 'compressed', f))
