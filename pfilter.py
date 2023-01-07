from os import listdir, path
import crepe
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm

base_dirs = ['D:/audio-datasety/MAN', 'D:/audio-datasety/WOMAN']

result = {}

for base_dir in base_dirs:
    speakers = listdir(base_dir)
    for i, speaker in enumerate(tqdm(speakers)):
        speaker_dir = path.join(base_dir, speaker)
        wavs_dir = path.join(speaker_dir, 'wavs')
        
        if not path.isdir(wavs_dir):
            continue

        meta_path = path.join(speaker_dir, 'metadata.csv')
        files = listdir(wavs_dir)
        if not path.exists(meta_path):
            continue

        with open(meta_path, encoding="utf-8") as reader:
            samples = list(map(lambda z: z[0],sorted([x.split('|') for _, x in enumerate(reader.readlines())], key=lambda x: x[4], reverse=True)))
        
        pitch_acc = 0
        qualified_samples = 0
        for file in tqdm(samples):
            if qualified_samples == 15:
                break
            current_path = path.join(wavs_dir, f'{file}.wav')
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
                if confidences[i] > 0.65:
                    qualified += 1
                    frequency_acc += frequencies[i]
                    qualified_frequencies.append(frequencies[i])
            #print(f'{file}: regions: {qualified}/{regions}, pitch: {frequency_acc / qualified} ({min(qualified_frequencies)} - {max(qualified_frequencies)})')
            if qualified > 0:
                pitch_acc += frequency_acc // qualified
                qualified_samples += 1
        result[speaker_dir] = pitch_acc // qualified_samples

with open('/outdir/summary2.txt', 'a', encoding="utf-8") as sf:
    sf.write(str(result))

        
