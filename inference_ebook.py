# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import os
import torch
import numpy as np
from torch.cuda import amp
from scipy.io.wavfile import write

from infer.helpers import load_models
import sys
sys.path.append('./infer')

# 0-Narrator
# 1-Will
# 2-Jenny
# 3-George
# 4-Horace
# 5-Alyss
# 6-Lady Pauline
# 7-Sir Rodney
# 8-Baron Arald
# 9-Sekretarz Martin
# 10-Skryba Nigel
# 11-Kuchmistrz Chubb
# 12-Mistrz koni Ulf
# 13-Zwiadowca Halt
# 14-Aida
# 15-Bryn
# 16-Jerome
# 17-Sir Karel
# 18-Starszy Kadet Paul
# 19-Sir Morton
# 20-Stary Bob

text_path = 'C:/model/ebook.txt'
output_dir = 'C:/outdir/audio4'

models_paths = [
    ('C:/outdir/models/radtts/shmart.pt', '', 'D:/colab/waveglow/25062022/shmart.pt', '', 'waveglow'),
    # ('C:/outdir/models/radtts/shmart.pt', '', 'c:/shmart/hifigan/cp_hifigan_p128/g_latest', 'c:/shmart/hifigan/cp_hifigan/config.json'),
    ('C:/outdir/models/radtts/w49p150.pt', '', 'c:/shmart/hifigan/cp_hifigan_woman/g543k', 'c:/shmart/hifigan/cp_hifigan_woman/config.json'),
]

speakers_map = {
    '0': ('M', 'shmart2', .2),
    '1': ('M', 'Geralt'),
    '2': ('F', 'Barbara Kałużna'),
    '3': ('M', 'Barnaba'),
    '4': ('M', 'Kharim'),
    '5': ('F', 'Cirilla'),
    '6': ('F', 'Karolina Gorczyca'),
    '7': ('M', 'Olgierd'),
    '8': ('M', 'Baron'),
    '9': ('M', 'ChamberlainEmhyr'),
    '10': ( 'M', 'Damien'),
    '11': ( 'M', 'Guy'),
    '12': ( 'M', 'Letho'),
    '13': ( 'M', 'Udalryk'),
    '14': ( 'M', 'Pazura'),
    '15': ( 'M', 'Zoltan'),
    '16': ( 'M', 'Nyras'),
    '17': ( 'M', 'Grimes'),
    '18': ( 'M', 'Raven'),
    '19': ( 'M', 'EwaldBorsody'),
    '20': ( 'M', 'Vesemir'),
}

def lines_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()

    lines = [f.rstrip() for f in lines]
    return lines


def infer():
    seed = 1337

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model1 = load_models(*models_paths[0])
    model2 = load_models(*models_paths[1])


    text_list = lines_to_list(text_path)

    os.makedirs(output_dir, exist_ok=True)
    for i, text in enumerate(text_list):
        suffix_path = f'{i+1:04}'
        output_path = f"{output_dir}/{suffix_path}.wav"

        if os.path.exists(output_path):
            continue

        speaker, text = text.split('|')

        model, _speaker_id, *infer_args = speakers_map[speaker]

        if model == 'S':
            continue
        radtts, vocode_audio, trainset, data_config = model1 if model == 'M' else model2

        speaker_id = trainset.get_speaker_id(_speaker_id).cuda()
        speaker_id_text, speaker_id_attributes = speaker_id, speaker_id
        
        if len(infer_args) > 0:
            sigma = infer_args[0]
        else:
            sigma = 0.667

        #print(f"{i+1}/{len(text_list)}: {text}")
        text = trainset.get_text(text).cuda()[None]
        with amp.autocast(False):
            with torch.no_grad():
                outputs = radtts.infer(
                    speaker_id, text, sigma, 0.667, 0.8,
                    0.8, 1.00, token_duration_max=100,
                    speaker_id_text=speaker_id_text,
                    speaker_id_attributes=speaker_id_attributes,
                    f0_mean=0, f0_std=0, energy_mean=0,
                    energy_std=0)

                mel = outputs['mel']
                audio_denoised = vocode_audio(mel)

                write(output_path, data_config['sampling_rate'], audio_denoised)


if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    infer()
