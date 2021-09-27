import os
import json
import math
import torch

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence

from scipy.io import wavfile
import numpy as np
import datetime

def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))

def get_text(text, hps):
    phones = text
    text_norm = cleaned_text_to_sequence(phones)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./configs/baker_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

# print(net_g) 输出模型结构
# print(net_g.state_dict())
# for name in net_g.state_dict():
#     print(name)

_ = utils.load_checkpoint("./logs/baker_base/G_270000.pth", net_g, None)

stn_tst = get_text("Happiness is a way station between too much and too little!", hps)

print(datetime.datetime.now())
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
print(datetime.datetime.now())

save_wav(audio, 'baker_tts.wav', hps.data.sampling_rate)

# 保存整个网络
#torch.save(net_g, 'test_model_full.pth') 
# 保存网络中的参数, 速度快，占空间少
#torch.save(net_g.state_dict(), 'test_model_data.pth')