import asyncio
import websockets
import json
import numpy as np
import torch
import os

from tqdm import tqdm
from torch.cuda import amp
from scipy.io.wavfile import write
from infer.helpers import load_models

from radtts import RADTTS
from data import Data

from subtitle_parser import parse_subtitles_file

#DEFAULTS, can be parameterized in future:
sigma_tkndur = 0.667
# sigma_f0 = 1.0
# sigma_energy = 1.0
# f0_mean = 0.0
f0_std = 0.0
energy_mean = 0.0
energy_std = 0.0
denoising_strength = 0.006
    
data_config = None
vocode_audio = None
model: RADTTS = None
trainset: Data = None

async def initialize_model(websocket, message):
    model_config_path = message['$modelConfigPath']
    model_path = message['$modelPath']
    vocoder_config_path = message['$vocoderConfigPath']
    vocoder_path = message['$vocoderPath']

    info_message = json.dumps({'info': f'Initializing model from path: {model_path}...'})
    await websocket.send(info_message)

    global vocode_audio, model, trainset, data_config
    model, vocode_audio, trainset, data_config = load_models(model_path, model_config_path, vocoder_path, vocoder_config_path)

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

            audio_denoised = vocode_audio(outputs['mel'])

            file_name = f"{speaker}_{token_dur_scaling}_{sigma}_{sigma_tkndur}.wav"
            output_path = f"{output_dir}/{file_name}"
            write(output_path, 22050, audio_denoised)

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

               
                audio_denoised = vocode_audio(outputs['mel'])
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

    speaker_id = trainset.get_speaker_id(speaker).cuda()
    speaker_id_text = trainset.get_speaker_id(speaker_text).cuda()
    speaker_id_attributes = trainset.get_speaker_id(speaker_attributes).cuda()
    
    os.makedirs(output_dir, exist_ok=True)
    subtitles = parse_subtitles_file(file)

    result = None
    sampling_rate = data_config['sampling_rate']
    _token_dur_scaling = token_dur_scaling

    for i, subtitle in enumerate(tqdm(subtitles)):
        text = trainset.get_text(subtitle[0]).cuda()[None]

        with amp.autocast(False):
            with torch.no_grad():
                if result is None or (type(result) == list and len(result) == 0):
                    segment_length = subtitle[2]
                    result = []
                else:
                    segment_length = subtitle[2] - result.shape[0] / sampling_rate
                try:
                    audio_length = float('inf')
                    while audio_length > segment_length:
                        #if _token_dur_scaling != token_dur_scaling: print(f'with token duration scaling = {_token_dur_scaling}')

                        outputs = model.infer(
                            speaker_id, text, sigma, sigma_tkndur, sigma_f0,
                            sigma_energy, _token_dur_scaling, token_duration_max=100,
                            speaker_id_text=speaker_id_text,
                            speaker_id_attributes=speaker_id_attributes,
                            f0_mean=f0_mean, f0_std=f0_std, energy_mean=energy_mean,
                            energy_std=energy_std)

                       
                        audio_denoised = vocode_audio(outputs['mel'])

                        audio_length = audio_denoised.shape[0] / sampling_rate
                        _token_dur_scaling = _token_dur_scaling - 0.04

                    audio_denoised = pad_audio(audio_denoised, sampling_rate, segment_length)
                    result = np.concatenate([result, audio_denoised])
                    _token_dur_scaling = token_dur_scaling
                except Exception as e:
                    print(e)
                    continue

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

               
                audio_denoised = vocode_audio(outputs['mel'])
                if result is None:
                    result = audio_denoised
                else:
                    result = np.concatenate([result, audio_denoised])

    file_name = f"{speaker}_{token_dur_scaling}_{sigma}_{sigma_tkndur}.wav"
    output_path = f"{output_dir}/{file_name}"
    write(output_path, 22050, result)
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