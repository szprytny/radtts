import auditok
import whisper
import os
import crepe
import timbral_models
import torch
from auditok import AudioRegion
from scipy.io import wavfile

def get_finetuned_model(model_name='small', fine_tuned_model=None):
  temp_model = whisper.load_model(model_name)

  if fine_tuned_model is None:
    return temp_model

  state_d_openai = temp_model.state_dict()
  state_dict_finetuned = torch.load(fine_tuned_model)['state_dict']

  new_state_dict = {}
  n_except = 0
  for k in state_d_openai.keys():
      try:
          new_state_dict[k] = state_dict_finetuned["model." + k]
      except:
          n_except += 1

  assert n_except == 0, "Rebuild state dict failed"

  temp_model.load_state_dict(new_state_dict)

  return temp_model


model = get_finetuned_model("medium")#, 'C:/Users/user/Downloads/whisper.ckpt') #)

options = whisper.DecodingOptions(language="pl", beam_size=5, without_timestamps=True)

def get_regions(input: AudioRegion, energy_threshold: float):
    return auditok.split(
        input,
        min_dur=1.62,     # minimum duration of a valid audio event in seconds
        max_dur=20.2,       # maximum duration of an event
        max_silence=0.4, # maximum duration of tolerated continuous silence within an event
        energy_threshold=energy_threshold # threshold of detection
    )

def find_threshold(input: AudioRegion):
    best = 0
    best_amount = 0
    x = 10
    while x < 100:
        audio_regions = get_regions(input, x)
        regions = 0
        for i in enumerate(audio_regions):
            regions += 1

        avg = input.duration / (regions or 1)

        if avg < 8.1:
            print(f'found threshold = {x}, regions: {regions}')
            return x

        if best_amount < regions:
            best_amount = regions
            best = x
        x += 2

    print(f'found threshold = {best}, regions: {best_amount}')
    return best

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

def split_and_transcribe(input_wav_path: str, output_path: str, basename: str, with_pitch=False, with_timbre=False, just_split=False) :
    if not os.path.exists(output_path): os.mkdir(output_path)

    transcript = open(f"{output_path}/metadata.csv", 'a', encoding="utf-8")
    input = auditok.load(input_wav_path)

    energy_threshold = find_threshold(input)

    audio_regions = get_regions(input, energy_threshold)

    regions = 0

    split_output_dir = f'{output_path}/wavs'
    if not os.path.exists(split_output_dir): os.mkdir(split_output_dir)

    print(f'current chapter length: {input.duration}')

    for i, r in enumerate(audio_regions):
        regions += 1

        split_template=f'{basename}_{i:04}'
        output_name = f"{split_output_dir}/{split_template}.wav"
        if os.path.exists(output_name):
            continue
        
        if(r.duration > 10.2 or r.duration < 1.62): continue

        print(f"\rRegion {i}: {r.meta.start:.3f}s -- {r.meta.end:.3f}s".ljust(30, ' '), end="")

        filename = r.save(output_name, audio_format="wav", exists_ok=True, audio_parameters={"sr":22050, "ch": 1})

        if just_split:
            continue

        if with_pitch:
          pitch_string = get_pitch(filename)

          if pitch_string is None:
              continue
        else:
          pitch_string='0|0|0'

        if with_timbre:
          timbre_string = get_timbre(filename)

          if timbre_string is None:
              continue
        else:
          timbre_string='0|0|0|0|0|0|0|0'

        audio = whisper.load_audio(filename, 22050)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        result = whisper.decode(model, mel, options)

        transcript.write(f'{split_template}|{result.text}|undefined|undefined|{r.duration}|{pitch_string}|{timbre_string}\n')

    avg = input.duration / (regions or 1)
    print(f'\rregions: {regions}, avg: {avg}'.ljust(30, ' '))
    transcript.close()

base_dir = 'C:/model/tosplit'
out_dir = 'c:/model/splitter2'
speaker_dirs = os.listdir(base_dir)

with_pitch=False
with_timbre=False
just_split=False
book_offset = 0
chapter_offset = 0

for i, speaker_dir in enumerate(speaker_dirs):
    speaker_books = os.listdir(f'{base_dir}/{speaker_dir}')
    for j, speaker_book in enumerate(speaker_books):
        chapters = os.listdir(f'{base_dir}/{speaker_dir}/{speaker_book}')
        for k, chapter in enumerate(chapters):
                current_chapter = f'{base_dir}/{speaker_dir}/{speaker_book}/{chapter}'
                print(f'working on {current_chapter} ({k+1}/{len(chapters)})')
                split_and_transcribe(current_chapter, f'{out_dir}/{speaker_dir}', f'{j + book_offset}_{k + chapter_offset}', with_pitch, with_timbre, just_split)