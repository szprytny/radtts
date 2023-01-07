import os
import crepe
from scipy.io import wavfile
import numpy as np

base_dir = 'C:\model\EXP2'
speakers = os.listdir(base_dir)
for i, speaker in enumerate(speakers):
    speaker_dir = f'{base_dir}/{speaker}'
    wavs_dir = f'{speaker_dir}/wavs'
    
    if not os.path.isdir(wavs_dir):
        continue

    files = os.listdir(wavs_dir)
    meta_path = f"{speaker_dir}/pitch.csv"
    if (os.path.exists(meta_path)):
        reader = open(meta_path)
        done = [x.split('|')[0] for _, x in enumerate(reader.readlines())]
        reader.close()
    else: done = []
    
    print(f'Working on {speaker_dir}')
    for k, file in enumerate(files):
        print(f'\r{k+1} / {len(files)}', end="")
        if file in done:
            continue
        meta_file = open(meta_path, 'a', encoding="utf-8")
        current_path = f'{wavs_dir}/{file}'
        sr, audio = wavfile.read(current_path)
        timestamps, frequencies, confidences, *_ = crepe.predict(audio, sr, model_capacity='full', verbose=0, step_size=5)
        regions = 0
        qualified = 0
        frequency_acc = 0
        frequencies = list(frequencies)
        qualified_frequencies = []
        confidences = list(confidences)

        for i, v in enumerate(timestamps):
            regions += 1
            if confidences[i] > 0.85:
                qualified += 1
                frequency_acc += frequencies[i]
                qualified_frequencies.append(frequencies[i])
        if len(qualified_frequencies) > 0:
            meta_file.write(f'{file}|{int(frequency_acc / (qualified or 1))}|{int(min(qualified_frequencies))}|{int(max(qualified_frequencies))}\n')
        meta_file.close()
        #print(f'{file}: regions: {qualified}/{regions}, pitch: {frequency_acc / qualified} ({min(qualified_frequencies)} - {max(qualified_frequencies)})')
