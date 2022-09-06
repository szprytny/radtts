import asyncio
import websockets
import json
import numpy as np
import torch
import os

from torch.cuda import amp
from scipy.io.wavfile import write

from radtts import RADTTS
from data import Data

from hifigan_models import Generator
from hifigan_env import AttrDict
from hifigan_denoiser import Denoiser

#DEFAULTS, can be parameterized in future:
sigma_tkndur = 0.667
# sigma_f0 = 1.0
# sigma_energy = 1.0
# f0_mean = 0.0
f0_std = 0.0
energy_mean = 0.0
energy_std = 0.0
denoising_strength = 0.006

vocoder: Generator = None
denoiser: Denoiser = None
data_config, model_config = None, None
model: RADTTS = None
trainset: Data = None

def load_vocoder(config_path, vocoder_path, to_cuda=True):
    with open(config_path) as f:
        data_vocoder = f.read()
    config_vocoder = json.loads(data_vocoder)
    h = AttrDict(config_vocoder)
    if 'gaussian_blur' in config_vocoder:
        config_vocoder['gaussian_blur']['p_blurring'] = 0.0
    else:
        config_vocoder['gaussian_blur'] = {'p_blurring': 0.0}
        h['gaussian_blur'] = {'p_blurring': 0.0}

    state_dict_g = torch.load(vocoder_path, map_location='cpu')['generator']

    # load hifigan
    global vocoder
    vocoder = Generator(h)
    vocoder.load_state_dict(state_dict_g)
    global denoiser
    denoiser = Denoiser(vocoder)
    if to_cuda:
        vocoder.cuda()
        denoiser.cuda()
    vocoder.eval()
    denoiser.eval()

def load_model(config_path: str, model_path: str):
    with open(config_path) as f:
        data = f.read()
    config = json.loads(data)
    
    global data_config
    data_config = config["data_config"]
    global model_config
    model_config = config["model_config"]

    global model
    model = RADTTS(**model_config).cuda()
    model.enable_inverse_cache() # cache inverse matrix for 1x1 invertible convs

    checkpoint_dict = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint_dict['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded checkpoint '{model_path}')")
    
    ignore_keys = ['training_files', 'validation_files']
    
    global trainset
    trainset = Data(
        data_config['training_files'],
        **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))

async def initialize_model(websocket, message):
    model_config_path = message['$modelConfigPath']
    model_path = message['$modelPath']
    vocoder_config_path = message['$vocoderConfigPath']
    vocoder_path = message['$vocoderPath']

    info_message = json.dumps({'info': f'Initializing model from path: {model_path}...'})
    await websocket.send(info_message)

    load_model(model_config_path, model_path)
    load_vocoder(vocoder_config_path, vocoder_path)

    info_message = json.dumps({
            'info': f'Initialized model from path: {model_path}...',
            'speakers': json.dumps(trainset.speaker_ids)
        })
    await websocket.send(info_message)

async def handle_request(websocket, message):
    if message['req'] == 'getSpeakers':
        if trainset is not None:
            info_message = json.dumps({
                'info': f'Returning list of speakers...',
                'speakers': json.dumps(trainset.speaker_ids)
            })
        else:
            info_message = json.dumps({
                'info': f'Model has not been initialized yet.'
            })

        await websocket.send(info_message)

async def synthesize_text(websocket, message):
    text = message['textToSynthesize']
    sigma = float(message['sigma'])
    speaker = message['speaker']
    speaker_attributes = message['speakerAttributes'] if 'speakerAttributes' in message else speaker
    speaker_text = message['speakerText'] if 'speakerText' in message else speaker
    output_dir = message['$outputPath']
    token_dur_scaling = message['speed']
    sigma_energy = message['energy']
    sigma_f0 = message['f0']
    f0_mean = message['f0Mean']

    speaker_id = trainset.get_speaker_id(speaker).cuda()
    speaker_id_text = trainset.get_speaker_id(speaker_text).cuda()
    speaker_id_attributes = trainset.get_speaker_id(speaker_attributes).cuda()
    
    os.makedirs(output_dir, exist_ok=True)
    text = trainset.get_text(text).cuda()[None]

    with amp.autocast(False):
        with torch.no_grad():
            outputs = model.infer(
                speaker_id, text, sigma, sigma_tkndur, sigma_f0,
                sigma_energy, token_dur_scaling, token_duration_max=100,
                speaker_id_text=speaker_id_text,
                speaker_id_attributes=speaker_id_attributes,
                f0_mean=f0_mean, f0_std=f0_std, energy_mean=energy_mean,
                energy_std=energy_std)

            mel = outputs['mel']
            audio = vocoder(mel).float()[0]
            audio_denoised = denoiser(audio, strength=denoising_strength)[0].float()

            audio = audio[0].cpu().numpy()
            audio_denoised = audio_denoised[0].cpu().numpy()
            audio_denoised = audio_denoised / np.max(np.abs(audio_denoised))

            file_name = f"{speaker}_{token_dur_scaling}_{sigma}_{sigma_tkndur}.wav"
            output_path = f"{output_dir}/{file_name}"
            write(output_path, data_config['sampling_rate'], audio_denoised)

    info_message = json.dumps({
        'info': f'Synthesized file...',
        'audioPath': output_path
    })
    await websocket.send(info_message)
    print(f"generated file {output_path}")


def lines_to_list(filename):
    with open(filename, encoding='utf-8') as f:
        texts = f.readlines()

    texts = [line.rstrip() for line in texts]
    return texts

async def synthesize_file(websocket, message):
    file = message['fileToSynthesize']
    sigma = float(message['sigma'])
    speaker = message['speaker']
    speaker_attributes = message['speakerAttributes'] if 'speakerAttributes' in message else speaker
    speaker_text = message['speakerText'] if 'speakerText' in message else speaker
    output_dir = message['$outputPath']
    token_dur_scaling = message['speed']
    sigma_energy = message['energy']
    sigma_f0 = message['f0']
    f0_mean = message['f0Mean']

    speaker_id = trainset.get_speaker_id(speaker).cuda()
    speaker_id_text = trainset.get_speaker_id(speaker_text).cuda()
    speaker_id_attributes = trainset.get_speaker_id(speaker_attributes).cuda()
    
    os.makedirs(output_dir, exist_ok=True)
    texts = lines_to_list(file)

    result = None

    for i, text in enumerate(texts):
        text = trainset.get_text(text).cuda()[None]

        with amp.autocast(False):
            with torch.no_grad():
                outputs = model.infer(
                    speaker_id, text, sigma, sigma_tkndur, sigma_f0,
                    sigma_energy, token_dur_scaling, token_duration_max=100,
                    speaker_id_text=speaker_id_text,
                    speaker_id_attributes=speaker_id_attributes,
                    f0_mean=f0_mean, f0_std=f0_std, energy_mean=energy_mean,
                    energy_std=energy_std)

                mel = outputs['mel']
                audio = vocoder(mel).float()[0]
                audio_denoised = denoiser(audio, strength=denoising_strength)[0].float()

                audio = audio[0].cpu().numpy()
                audio_denoised = audio_denoised[0].cpu().numpy()
                audio_denoised = audio_denoised / np.max(np.abs(audio_denoised))
                if result is None:
                    result = audio_denoised
                else:
                    result = np.concatenate([result, audio_denoised])

    file_name = f"{speaker}_{token_dur_scaling}_{sigma}_{sigma_tkndur}.wav"
    output_path = f"{output_dir}/{file_name}"
    write(output_path, data_config['sampling_rate'], result)
    info_message = json.dumps({
        'info': f'Synthesized text...',
        'audioPath': output_path
    })
    await websocket.send(info_message)
    print(f"generated file {output_path}")

def pad_audio(data, fs, T):
    # Calculate target number of samples
    N_tar = int(fs * T)
    # Calculate number of zero samples to append
    shape = data.shape
    # Create the target shape    
    N_pad = N_tar - shape[0]
    shape = (N_pad,) + shape[1:]
    # Stack only if there is something to append    
    if shape[0] > 0:                
        if len(shape) > 1:
            return np.vstack((np.zeros(shape),
                              data))
        else:
            return np.hstack((np.zeros(shape),
                              data))
    else:
        return data

async def synthesize_subs(websocket, message):
    file = message['subsToSynthesize']
    sigma = float(message['sigma'])
    speaker = message['speaker']
    speaker_attributes = message['speakerAttributes'] if 'speakerAttributes' in message else speaker
    speaker_text = message['speakerText'] if 'speakerText' in message else speaker
    output_dir = message['$outputPath']
    token_dur_scaling = message['speed']
    sigma_energy = message['energy']
    sigma_f0 = message['f0']
    f0_mean = message['f0Mean']
    frame_rate = message['frameRate']

    speaker_id = trainset.get_speaker_id(speaker).cuda()
    speaker_id_text = trainset.get_speaker_id(speaker_text).cuda()
    speaker_id_attributes = trainset.get_speaker_id(speaker_attributes).cuda()
    
    os.makedirs(output_dir, exist_ok=True)
    texts = lines_to_list(file)

    result = None
    sampling_rate = data_config['sampling_rate']
    _token_dur_scaling = token_dur_scaling

    for i, text in enumerate(texts):
        from_frame, to_frame, text = text.split('|')
        text = trainset.get_text(text).cuda()[None]

        with amp.autocast(False):
            with torch.no_grad():
                if result is None:
                    segment_length = float(to_frame) / float(frame_rate)
                    result = []
                else:
                    segment_length = float(to_frame) / float(frame_rate) - (result.shape[0] / sampling_rate)

                audio_length = float('inf')
                while audio_length > segment_length:
                    if _token_dur_scaling != token_dur_scaling: print(f'with token duration scaling = {_token_dur_scaling}')

                    outputs = model.infer(
                        speaker_id, text, sigma, sigma_tkndur, sigma_f0,
                        sigma_energy, _token_dur_scaling, token_duration_max=100,
                        speaker_id_text=speaker_id_text,
                        speaker_id_attributes=speaker_id_attributes,
                        f0_mean=f0_mean, f0_std=f0_std, energy_mean=energy_mean,
                        energy_std=energy_std)

                    mel = outputs['mel']
                    audio = vocoder(mel).float()[0]
                    audio_denoised = denoiser(audio, strength=denoising_strength)[0].float()

                    audio = audio[0].cpu().numpy()
                    audio_denoised = audio_denoised[0].cpu().numpy()
                    audio_denoised = audio_denoised / np.max(np.abs(audio_denoised))

                    audio_length = audio_denoised.shape[0] / sampling_rate
                    _token_dur_scaling = _token_dur_scaling - 0.015

                audio_denoised = pad_audio(audio_denoised, sampling_rate, segment_length)
                result = np.concatenate([result, audio_denoised])
                _token_dur_scaling = token_dur_scaling

    file_name = f"{speaker}_subs.wav"
    output_path = f"{output_dir}/{file_name}"
    write(output_path, sampling_rate, result)
    info_message = json.dumps({
        'info': f'Synthesized subs...',
        'audioPath': output_path
    })
    await websocket.send(info_message)
    print(f"generated file {output_path}")

async def synthesize_book(websocket, message):
    file = message['bookToSynthesize']
    sigma = float(message['sigma'])
    output_dir = message['$outputPath']
    token_dur_scaling = message['speed']
    sigma_energy = message['energy']
    sigma_f0 = message['f0']
    f0_mean = message['f0Mean']
    
    os.makedirs(output_dir, exist_ok=True)
    texts = lines_to_list(file)

    result = None

    for i, text in enumerate(texts):
        
        speaker, text = text.split('|')

        speaker_id = torch.LongTensor([int(speaker)]).cuda()
        speaker_id_text = speaker_id
        speaker_id_attributes = speaker_id
        text = trainset.get_text(text).cuda()[None]

        with amp.autocast(False):
            with torch.no_grad():
                outputs = model.infer(
                    speaker_id, text, sigma, sigma_tkndur, sigma_f0,
                    sigma_energy, token_dur_scaling, token_duration_max=100,
                    speaker_id_text=speaker_id_text,
                    speaker_id_attributes=speaker_id_attributes,
                    f0_mean=f0_mean, f0_std=f0_std, energy_mean=energy_mean,
                    energy_std=energy_std)

                mel = outputs['mel']
                audio = vocoder(mel).float()[0]
                audio_denoised = denoiser(audio, strength=denoising_strength)[0].float()

                audio = audio[0].cpu().numpy()
                audio_denoised = audio_denoised[0].cpu().numpy()
                audio_denoised = audio_denoised / np.max(np.abs(audio_denoised))
                if result is None:
                    result = audio_denoised
                else:
                    result = np.concatenate([result, audio_denoised])

    file_name = f"{speaker}_{token_dur_scaling}_{sigma}_{sigma_tkndur}.wav"
    output_path = f"{output_dir}/{file_name}"
    write(output_path, data_config['sampling_rate'], result)
    info_message = json.dumps({
        'info': f'Synthesized book...',
        'audioPath': output_path
    })
    await websocket.send(info_message)
    print(f"generated file {output_path}")

async def echo(websocket):
    async for event in websocket:
        try:
            message = json.loads(event)
            if 'req' in message:
                await handle_request(websocket, message)
            if '$modelPath' in message:
                await initialize_model(websocket, message)
            if 'textToSynthesize' in message:
                await synthesize_text(websocket, message)
            if 'fileToSynthesize' in message:
                await synthesize_file(websocket, message)
            if 'subsToSynthesize' in message:
                await synthesize_subs(websocket, message)
            if 'bookToSynthesize' in message:
                await synthesize_book(websocket, message)
                
        except Exception as e: 
            await websocket.send(json.dumps({'info': f'Error...\n{e}'}))

async def main():
    seed = 4213
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    async with websockets.serve(echo, "localhost", seed):
        await asyncio.Future()  # run forever

asyncio.run(main())