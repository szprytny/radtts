import os
import argparse
import torch
from data import Data

from infer.helpers import get_config, ignore_keys

def enrich(checkpoint_path: str, config_path: str, out_path: str = None):
  loaded_obj = torch.load(checkpoint_path, map_location='cpu')

  config = loaded_obj['config'] if 'config' in loaded_obj else None
  speaker_ids = loaded_obj['speaker_ids'] if 'speaker_ids' in loaded_obj else None

  if config is not None and speaker_ids is not None or (out_path is not None and os.path.exists(out_path)):
    print(f'model {checkpoint_path} is already enriched.')
    return
  
  if config is None:
    config = get_config(config_path, {})
    loaded_obj['config'] = config

  if speaker_ids is None:
    data_config = config['data_config']
    trainset = Data(
      data_config['training_files'],
      **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))

    loaded_obj['speaker_ids'] = trainset.speaker_ids

  torch.save(loaded_obj, checkpoint_path.replace('.pt', '.ptsh') if out_path is None else out_path)

if __name__ == "__main__":
  models = [
    ('/outdir/models/skyrim1/model_17000', '/outdir/models/skyrim1/config.json', '/outdir/models/radtts/skyrim.pt'),
  ]

  for x in models:
    enrich(*x)
