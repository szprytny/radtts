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

from scipy.io.wavfile import write

import torch
from torch.cuda import amp
from infer.helpers import get_configs, load_models, ignore_keys

from torch.utils.data import DataLoader
from data import Data, DataCollate
from train import parse_data_from_batch



output_dir = 'C:/outdir/audio2'

models_paths = [
    ('D:/colab/RAD/220920/step2/51k.pt', 'D:/colab/RAD/220920/step2/config.json', 'c:/shmart/hifigan/cp_hifigan_man/g_shmart', 'c:/shmart/hifigan/cp_hifigan_man/config.json'),
    ('C:/outdir/models/witches/step2/model_45000', 'C:/outdir/models/witches/step2//config.json', 'c:/shmart/hifigan/cp_hifigan/g_latest', 'c:/shmart/hifigan/cp_hifigan/config.json'),
]

speaker_id = 'Bezi-normal'

def infer():
    seed = 1337

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    radtts, vocoder, denoiser, trainset = load_models(*models_paths[0])
    model_config, data_config = get_configs(models_paths[0][1], [])

    data_config['validation_files'] = {
        "LJS": {
            "basedir": "C:/todo/splitted/shmart",
            "audiodir": "C:/todo/splitted/shmart",
            "filelist": "test.txt",
            "lmdbpath": ""
        }
    }

    valset = Data(data_config['validation_files'],
                  **dict((k, v) for k, v in data_config.items()
                  if k not in ignore_keys),
                  speaker_ids=trainset.speaker_ids)
    collate_fn = DataCollate()
    dataloader = DataLoader(valset, num_workers=1, shuffle=False,
                            sampler=None, batch_size=1,
                            pin_memory=False, drop_last=False,
                            collate_fn=collate_fn)

    for i, batch in enumerate(dataloader):
        suffix_path = f'{i:04}'
        output_path = f"{output_dir}/{suffix_path}.wav"

        if os.path.exists(output_path):
            continue
            
        (mel, speaker_ids, text, in_lens, out_lens, attn_prior,
            f0, voiced_mask, p_voiced, energy_avg,
            audiopaths) = parse_data_from_batch(batch)
        filename = os.path.splitext(
            os.path.basename(audiopaths[0]))[0]
        f0_gt, energy_avg_gt = f0.clone(), energy_avg.clone()

        speaker=valset.get_speaker_id(speaker_id).cuda()

        print("sample", i, filename)
        with amp.autocast(False):
            # extract duration from attention using ground truth mel
            outputs = radtts(
                mel, speaker, text, in_lens, out_lens, True,
                attn_prior=attn_prior, f0=f0, energy_avg=energy_avg,
                voiced_mask=voiced_mask, p_voiced=p_voiced)

            dur_target = outputs['attn'][0, 0].sum(0, keepdim=True)

        dur_target = (dur_target + 0.5).floor().int()

        with amp.autocast(False):

            model_output = radtts.infer(
                speaker, text, 0.75, dur=dur_target, f0=f0,
                energy_avg=energy_avg, voiced_mask=voiced_mask,
                f0_mean=0, f0_std=0,
                energy_mean=0, energy_std=0)

            mel = model_output['mel']


            audio = vocoder(mel).float()[0]
            audio_denoised = denoiser(
                audio, strength=0.01)[0].float()
            audio = audio[0].cpu().numpy()
            audio_denoised = audio_denoised[0].cpu().numpy()

            write(output_path ,22050, audio_denoised)

if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    with torch.no_grad():
        infer()

