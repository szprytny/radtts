import torch
import json

from hifigan_models import Generator
from hifigan_env import AttrDict
from hifigan_denoiser import Denoiser

from radtts import RADTTS
from data import Data
from common import update_params

ignore_keys = ['training_files', 'validation_files']

def get_config(config_path, params):
    with open(config_path) as f:
        data = f.read()

    config = json.loads(data)
    update_params(config, params)

    return config

def get_configs(config_path, params):
    with open(config_path) as f:
        data = f.read()

    config = json.loads(data)
    update_params(config, params)

    data_config = config["data_config"]
    model_config = config["model_config"]

    return model_config, data_config


def load_vocoder(vocoder_path, config_path, to_cuda=True):
    with open(config_path) as f:
        data_vocoder = f.read()
    config_vocoder = json.loads(data_vocoder)
    h = AttrDict(config_vocoder)
    if 'blur' in vocoder_path:
        config_vocoder['gaussian_blur']['p_blurring'] = 0.5
    else:
        if 'gaussian_blur' in config_vocoder:
            config_vocoder['gaussian_blur']['p_blurring'] = 0.0
        else:
            config_vocoder['gaussian_blur'] = {'p_blurring': 0.0}
            h['gaussian_blur'] = {'p_blurring': 0.0}

    state_dict_g = torch.load(vocoder_path, map_location='cpu')['generator']

    # load hifigan
    vocoder = Generator(h)
    vocoder.load_state_dict(state_dict_g)
    denoiser = Denoiser(vocoder)
    if to_cuda:
        vocoder.cuda()
        denoiser.cuda()
    vocoder.eval()
    denoiser.eval()

    return vocoder, denoiser


def load_models(radtts_path, config_path, vocoder_path, vocoder_config_path):
    checkpoint_dict = torch.load(radtts_path, map_location='cpu')
    speaker_ids = checkpoint_dict['speaker_ids'] if 'speaker_ids' in checkpoint_dict else None
    
    config = checkpoint_dict['config'] if 'config' in checkpoint_dict else get_config(config_path)
    model_config = config['model_config']
    data_config = config['data_config']

    radtts = RADTTS(**model_config).cuda()
    radtts.enable_inverse_cache() # cache inverse matrix for 1x1 invertible convs

    state_dict = checkpoint_dict['state_dict']
    radtts.load_state_dict(state_dict, strict=False)
    radtts.eval()
    
    print(f"Loaded checkpoint '{radtts_path}')")

    vocoder, denoiser = load_vocoder(vocoder_path, vocoder_config_path)
    trainset = Data(
        data_config['training_files'],
        **dict((k, v) for k, v in data_config.items() if k not in ignore_keys),
        speaker_ids=speaker_ids,
        load_data=False)
    
    return radtts, vocoder, denoiser, trainset