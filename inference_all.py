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


text_path = 'C:/outdir/sentences.txt'
output_dir = 'C:/outdir/audio4'

models_paths = [
    ('C:/outdir/models/radtts/woman43.pt', '<<NOT NEEDED>>', 'c:/shmart/hifigan/cp_hifigan_woman/g543k', 'c:/shmart/hifigan/cp_hifigan/config.json'),
]

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

    model = load_models(*models_paths[0])

    text_list = lines_to_list(text_path)

    os.makedirs(output_dir, exist_ok=True)
    for i, text in enumerate(text_list):

        radtts, vocode_audio, trainset, data_config = model
        
        for j, speaker in enumerate(trainset.speaker_ids):
            suffix_path = f'{i+1:04}_{j}'
            output_path = f"{output_dir}/{suffix_path}.wav"

            if os.path.exists(output_path):
                continue
            speaker_id = trainset.get_speaker_id(speaker).cuda()
            speaker_id_text, speaker_id_attributes = speaker_id, speaker_id
        
        
            #print(f"{i+1}/{len(text_list)}: {text}")
            text_tensor = trainset.get_text(text).cuda()[None]
            with amp.autocast(False):
                with torch.no_grad():
                    outputs = radtts.infer(
                        speaker_id, text_tensor, 0.667, 0.667, 0.8,
                        0.8, 1.00, token_duration_max=100,
                        speaker_id_text=speaker_id_text,
                        speaker_id_attributes=speaker_id_attributes,
                        f0_mean=0, f0_std=0, energy_mean=0,
                        energy_std=0)

                    mel = outputs['mel']
                    audio = vocode_audio(mel)

                    write(output_path, data_config['sampling_rate'], audio)


if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    infer()

