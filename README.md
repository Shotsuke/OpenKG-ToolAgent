# DeepKE-mcp-tools

为 `DeepKE` 的四个自然语言处理任务的 **standard** 的 **预测** 部分的 mcp 服务，因此前提需要有 [DeepKE](https://github.com/zjunlp/DeepKE) 对应任务训练好的模型，确保对应的 `predict.py` 能够运行。

## 下载代码

```bash
cd DeepKE
git clone https://github.com/Shotsuke/deepke-mcp-tools.git
```

## 配置 `.env` 环境变量

- 配置deepke / deepke-ee虚拟环境
- 如果在训练模型时已配好环境，那么可以跳过这一部分。

```bash
cd DeepKE
conda create -n deepke python=3.8 -y
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

配置完deepke后再来配置deepke-ee，基本是相同的，不过多加一条 `pip install hydra-core==1.3.1` ， `EE` 任务使用版本更高的 `hydra` 。

将 `conda` 对应的包含 `PY` 的目录和 `DeepKE` 目录放入 `.env` 中，例如：

```
DEEPKE_PATH="~/DeepKE"
CONDA_PY="/home/user_name/anaconda3/envs/deepke/bin/"
CONDA_EE_PY="/home/user_name/anaconda3/envs/deepke-ee/bin/"
```

- API_KEY

使用了阿里的qwen大模型，更改 `DASHSCOPE_API_KEY` 即可

## 配置mcp项目uv环境

```bash
# curl -LsSf https://astral.sh/uv/install.sh | sh # 安装uv
# pipx install uv # 反正选一个安装uv
pip install uv

# Now in: DeepKE/
cd deepke-mcp-tools
uv venv
source .venv/bin/activate
uv add "mcp[cli]" httpx openai pyyaml
```

## 运行

```bash
# Now in: DeepKE/deepke-mcp-tools/
python run.py
```