# DeepKE-mcp-tools

## 1. 简介

`DeepKE` 是一个开源的知识图谱抽取与构建工具，支持**cnSchema、低资源、长篇章、多模态**的知识抽取工具，提供标准化的 NLP 抽取任务，包括：

- ✅ 命名实体识别 (NER)
- ✅ 关系抽取 (RE)
- ✅ 属性抽取 (AE)
- ✅ 事件抽取 (EE)

`DeepKE-mcp-tools` 在 `DeepKE` 原有任务的基础上，提供标准化、可独立调用的 MCP 服务接口，为四个任务的**常规全监督standatd**、**预测**部分提供了可供大语言模型调用的 mcp 服务。

> 💡 参考：关于 DeepKE 的完整介绍和训练指导，请参考 [DeepKE官方仓库](https://github.com/zjunlp/DeepKE)。

## 2. 使用及配置

> 在使用前，请先准备好对应任务已经训练好的模型，并确认 DeepKE 中的 predict.py 可正常调用。
> 
> 如果您希望先简单验证 MCP 服务是否正常，可修改 `DeepKE/example/ner/standard/predict.py` 中的内容，例如
> ```
> print("hello world! this is only for test, not real code")
> ``` 
>，并且引导大语言模型来测试。

### 下载代码

```bash
cd DeepKE
git clone https://github.com/Shotsuke/deepke-mcp-tools.git
```

### 配置 `.env` 环境变量

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

使用了阿里的qwen大模型，更改 `DASHSCOPE_API_KEY` 即可，用于演示。如果只需要将 mcp 服务嵌入到应用程序（如 Cursor ）则、不需要。

### 配置mcp项目uv环境

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

### 测试运行

相当于演示 demo ，也可以在这里直接对话。

```bash
# Now in: DeepKE/deepke-mcp-tools/
python run.py
```

### 在 Cursor & Cline 中使用

修改 `.cursor/mcp.json` 或 `cline_mcp_settings.json` ，或者在其他支持 mcp 服务的应用中添加类似的配置。

```json
{
  "mcpServers": {
    "DeepKE": {
      "command": "uv",
      "args": [
        "--directory",
        "absolute/path/to/DeepKE/deepke-mcp-tools/tools",
        "run",
        "server.py"
      ]
    }
  }
}
```

## 3. 支持的能力

`DeepKE-mcp-tools` 当前提供以下任务接口：

| 任务类型 | 接口名         | 功能概述                         |
| -------- | -------------- | -------------------------------- |
| NER      | `deepke_ner()` | 常规全监督命名实体识别预测       |
| RE       | `deepke_re()`  | 常规全监督关系抽取预测           |
| AE       | `deepke_ae()`  | 常规全监督属性抽取预测           |
| EE       | `deepke_ee()`  | 常规全监督事件检测及论元提取预测 |

## 4. 注意事项

- 该 mcp 服务目前仍然处于早期阶段，相关能力正在完善中。
- 当前默认适配 Linux 路径，Windows 需修改对应路径。