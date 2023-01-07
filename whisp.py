import whisper
import os
import crepe
import timbral_models
import torch
from scipy.io import wavfile
from tqdm import tqdm

model = whisper.load_model("medium")

options = whisper.DecodingOptions(language="pl", beam_size=5, without_timestamps=True)

def get_pitch(filename: str):
    sr, audio = wavfile.read(filename)
    timestamps, frequencies, confidences, *_ = crepe.predict(audio, sr, model_capacity='full', verbose=0, step_size=5)
    regions = 0
    qualified = 0
    frequency_acc = 0
    frequencies = list(frequencies)
    qualified_frequencies = []
    confidences = list(confidences)

    for i, v in enumerate(timestamps):
        regions += 1
        if confidences[i] > 0.875:
            qualified += 1
            frequency_acc += frequencies[i]
            qualified_frequencies.append(frequencies[i])

    if len(qualified_frequencies) > 0:
        return f'{int(frequency_acc / (qualified or 1))}|{int(min(qualified_frequencies))}|{int(max(qualified_frequencies))}'
    return None

def get_timbre(filename: str):
    timbre_data = None
    try:
        timbre = timbral_models.timbral_extractor(filename, output_type='list', verbose=False, clip_output=True)
        # [hardness, depth, brightness, roughness, warmth, sharpness, boominess, reverb]
        timbre_data = '|'.join(map(lambda x: str(x), timbre))
    finally:
        return timbre_data

def transcribe(filename: str, base_dir: str, with_pitch=False, with_timbre=False) :
    if not filename.endswith('.wav'):
        return
    f_name = filename.replace('.wav', '')
    transcript = open(f"{base_dir}/metadata.csv", 'a', encoding="utf-8")
    with open(f"{base_dir}/metadata.csv", 'r', encoding="utf-8") as t: 
        if f_name in list(map(lambda l: l.split('|')[0],t.readlines())):
            return
    if with_pitch:
        pitch_string = get_pitch(filename)

        if pitch_string is None:
            return
    else:
        pitch_string='0|0|0'

    if with_timbre:
        timbre_string = get_timbre(filename)

        if timbre_string is None:
            return
    else:
        timbre_string='0|0|0|0|0|0|0|0'

    audio = whisper.load_audio(filename, 22050)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    result = whisper.decode(model, mel, options)

    transcript.write(f'{f_name}|{result.text}|undefined|undefined|-1|{pitch_string}|{timbre_string}\n')
    transcript.close()

base_dir = 'C:/model/todo'
speaker_dirs = os.listdir(base_dir)

with_pitch=False
with_timbre=False

for i, dataset_dir in enumerate(speaker_dirs):
    speaker_dirs = os.listdir(f'{base_dir}/{dataset_dir}')
    for j, speaker_dir in enumerate(speaker_dirs):
        speaker_files = os.listdir(f'{base_dir}/{dataset_dir}/{speaker_dir}')
        for k, speaker_file in enumerate(tqdm(speaker_files)):
                current_file = f'{base_dir}/{dataset_dir}/{speaker_dir}/{speaker_file}'
                #print(f'working on {current_file} ({k+1}/{len(speaker_files)})')
                transcribe(current_file, f'{base_dir}/{dataset_dir}/{speaker_dir}', with_pitch, with_timbre)