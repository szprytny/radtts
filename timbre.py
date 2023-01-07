import os
import timbral_models
from multiprocessing import Pool

def get_file_timbre_wrapper(args):
    return get_file_timbre(*args)

def get_file_timbre(wavs_dir: str, file: str):
    try:
        current_path = f'{wavs_dir}/{file}'
        timbre = timbral_models.timbral_extractor(current_path, output_type='list', verbose=False, clip_output=True)
        # [hardness, depth, brightness, roughness, warmth, sharpness, boominess, reverb]
        timbre_data = '|'.join(map(lambda x: str(x), timbre))
        return f'{file}|{timbre_data}\n'
    except:
        return ''

def to_chunks(list, chunk_size):
    return [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]

if __name__ == '__main__':
    base_dir = 'C:/model/EXP2'
    speakers = os.listdir(base_dir)
    for i, speaker in enumerate(speakers):
        speaker_dir = f'{base_dir}/{speaker}'
        wavs_dir = f'{speaker_dir}/wavs'
        
        if not os.path.isdir(wavs_dir):
            continue



        files = os.listdir(wavs_dir)
        meta_dir = f"{speaker_dir}/timbre.csv"
        if (os.path.exists(meta_dir)):
            reader = open(meta_dir)
            done = [x.split('|')[0] for _, x in enumerate(reader.readlines())]
            reader.close()
        else: done = []
        
        print(f'Working on {speaker_dir}')

        files_to_process = []
        for k, file in enumerate(files):
            if file in done:
                continue
            files_to_process.append(file)

        batch_size = 10
        chunk_size = 4 * batch_size

        chunks = to_chunks(files_to_process, chunk_size)

        for k, chunk in enumerate(chunks):
            print(f'\rprocessing chunk {k+1} of {len(chunks)}', end='')
            pool = Pool(batch_size)
            lines = pool.map(get_file_timbre_wrapper, [(wavs_dir, file) for file in chunk])
            pool.close()
            pool.join()
            
            meta_file = open(meta_dir, 'a', encoding="utf-8")
            for j, line in enumerate(lines):
                meta_file.write(line)
            meta_file.close()

        print('\n')