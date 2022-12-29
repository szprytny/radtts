import argparse
import json
from torch.utils.data import DataLoader
from data import Data, DataCollate
from common import update_params

def prepare_dataloaders(data_config):
    # Get data, data loaders and collate function ready
    ignore_keys = ['training_files', 'validation_files']
    print("initializing training dataloader")
    trainset = Data(data_config['training_files'],
                    **dict((k, v) for k, v in data_config.items()
                    if k not in ignore_keys))

    print(f'numebr of speakers: {len(trainset.speaker_ids)}')

    print("initializing validation dataloader")
    data_config_val = data_config.copy()
    data_config_val['aug_probabilities'] = None  # no aug in val set
    valset = Data(data_config['validation_files'],
                  **dict((k, v) for k, v in data_config_val.items()
                  if k not in ignore_keys), speaker_ids=trainset.speaker_ids)

    collate_fn = DataCollate(**data_config)

    train_sampler, shuffle = None, True

    train_loader = DataLoader(trainset, num_workers=4, shuffle=shuffle,
                              sampler=train_sampler, batch_size=1,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)
                        
    
    val_loader = DataLoader(valset, sampler=None, num_workers=4,
                            shuffle=False, batch_size=1,
                            pin_memory=False, collate_fn=collate_fn)

    return train_loader, val_loader


def process():

    train_loader, val_loader = prepare_dataloaders(
        data_config)

    for _ in train_loader:
        continue

    for _ in val_loader:
        continue

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)

    update_params(config, args.params)
    print(config)

    global data_config
    data_config = config["data_config"]

    process()
