## 配置`.env`环境变量

- 配置deepke / deepke-ee虚拟环境

```bash
cd DeepKE
conda create -n deepke python=3.8
conda activate deepke

pip install pip==24.0.0 # 要求 pip<=24.0
pip install -r requirements.txt
pip install -U transformers==4.36.2 # 有若干库指定`transformers == 3.4.0`这个版本，但实际上没法运行

# `requirements.txt`不会检查`torch`和`cuda`版本，因此需要手动检查
# conda list | grep "torch"
# nvidia-smi
# 根据`nvidia`适合的`cuda`来到([Start Locally | PyTorch](https://pytorch.org/get-started/locally/))选择合适的版本下载，例如：
pip install torch==2.4.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python setup.py install
python setup.py develop
```

配置完deepke后再来配置deepke-ee，基本是相同的，不过多加一条`pip install hydra-core==1.3.1`，EE使用版本更高的`hydra`。

将conda对应的包含PY的目录和DeepKE目录放入.env中，例如：

```
DEEPKE_PATH="~/DeepKE"
CONDA_PY="/home/user_name/anaconda3/envs/deepke/bin/"
CONDA_EE_PY="/home/user_name/anaconda3/envs/deepke-ee/bin/"
```

- API_KEY

使用了ali的qwen大模型，更改`DASHSCOPE_API_KEY`即可

## 配置mcp项目uv环境

```bash
# curl -LsSf https://astral.sh/uv/install.sh | sh # 安装uv
# pipx install uv # 反正选一个安装uv
pip install uv

cd DeepKE/mcp-tools
uv venv
source .venv/bin/activate
uv add "mcp[cli]" httpx openai pyyaml
```