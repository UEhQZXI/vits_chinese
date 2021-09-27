# vits_baker
vits实现的中文TTS

this is the copy of https://github.com/jaywalnut310/vits

VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

如果有侵权行为，请联系我，我将删除项目
If there is infringement, please contact me and I will delete the item

# 基于VITS 实现 16K baker TTS 的流程记录
# 核心在前端文本处理 

apt-get install espeak
pip install -r requirements.txt

# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

# 使用MFA对齐，使用TensorflowTTS提取train.txt即完成预处理
# 需要修改utils.py里面的数据路径
# dataPath = './baker_waves/'
# dataPost = '.wav'
# 弃用上面的方法，是用preprocess.py将MFA格式转换为VITS格式
python train.py -c configs/baker_base.json -m baker_base

# 上面的模型训练出来后存在，明显停顿的问题
# 原因：
# 1，本来是用MFA已经在音素后面强插边界了，VITS又强插边界了，具体是配置参数："add_blank": true
# 2，可能影响，随机时长预测，具体配置参数：use_sdp=True,
