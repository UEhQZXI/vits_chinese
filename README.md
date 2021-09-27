vits实现的中文TTS

this is the copy of https://github.com/jaywalnut310/vits		

VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech		

Espnet连接：github.com/espnet/espnet/tree/master/espnet2/gan_tts/vits

coqui-ai/TTS连接：github.com/coqui-ai/TTS/tree/main/recipes/ljspeech/vits_tts

如果有侵权行为，请联系我，我将删除项目

If there is infringement, please contact me and I will delete the item

# 基于VITS 实现 16K baker TTS 的流程记录

apt-get install espeak

pip install -r requirements.txt

cd monotonic_align

python setup.py build_ext --inplace

# 将16K标贝音频拷贝到./baker_waves/，启动训练

python train.py -c configs/baker_base.json -m baker_base

两张1080卡，训练两天，基本可以使用了

# 测试
python vits_strings.py

上面的模型训练出来后存在，明显停顿的问题

原因：	

1，本来已经在音素后面强插边界了，VITS又强插边界了，具体是配置参数："add_blank": true

2，可能影响，随机时长预测，具体配置参数：use_sdp=True

