# OpenKG-ToolAgent

## 1. 简介

### [OpenKG](http://openkg.cn/)

**OpenKG**（开放知识图谱）是由中国中文信息学会语言与知识计算专业委员会于2015年发起的公益性、学术性、开放性的知识图谱社区项目，旨在推动以中文为核心的知识图谱数据的开放、互联与众包，并促进知识图谱算法、工具及平台的开源开放。

### 本仓库

**OpenKG-ToolAgent** 是基于 **OpenKG 生态** 的知识图谱工具集成平台，旨在为知识图谱抽取、表示学习及下游推理任务提供 统一、标准化、可远程调用的 MCP 服务接口。您可以仅将服务端相关代码运行供您的 AI 程序调用，也可以使用我们给出的单次的思维链调用。

#### DeepKE

`DeepKE` 是一个开源的知识图谱抽取与构建工具，支持**cnSchema、低资源、长篇章、多模态**的知识抽取工具，提供标准化的 NLP 抽取任务，包括：

- ✅ 命名实体识别 (NER)
- ✅ 关系抽取 (RE)
- ✅ 属性抽取 (AE)
- ✅ 事件抽取 (EE)

`OpenKG-ToolAgent` 在 `DeepKE` 原有任务的基础上，提供标准化、可独立调用的 MCP 服务接口，为四个任务的**常规全监督standatd**、**预测**部分提供了可供大语言模型调用的 MCP 服务。

> 💡 参考：关于 DeepKE 的完整介绍和训练指导，请参考 [DeepKE官方仓库](https://github.com/zjunlp/DeepKE)。

#### muKG

`muKG` 是由南京大学 Websoft 研究组开发的一个 开源 Python 库，用于对 **单源或多源知识图谱** 进行表示学习。支持多种嵌入任务，包括： 

- ✅ 链路预测 (LP)
- ✅ 实体对齐 (EA)
- ✅ 实体类型识别 (ET)

同样地，本项目也为其实现了 MCP 服务，**包括训练和预测**。

> 💡 参考：关于 muKG 的完整介绍和训练指导，请参考 [muKG官方仓库](https://github.com/nju-websoft/muKG)。

## 2. 使用及配置

> **DeepKE**
> 
> 在使用前，请先准备好对应任务已经训练好的模型，并确认 DeepKE 中的 predict.py 可正常调用。
> 
> 如果您希望先简单验证 MCP 服务是否正常，可修改 `DeepKE/example/ner/standard/predict.py` 中的内容，例如
> ```
> print("hello world! this is only for test, not real code")
> ``` 
>，并且引导大语言模型来测试。

> **muKG**
> 
> 考虑到这部分任务的输入输出并不适合直接阅读，因此训练模型和提前训练模型来测试都相当重要。本仓库一并撰写了训练和预测的调用接口。输出的模型存储在 `MUKG_PATH/output` 下，当然您也可以自备训练好的模型。

### DeepKE 配置

```bash
git clone https://github.com/zjunlp/DeepKE.git
```

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

**写入 `.env` 文件**

将 `conda` 对应的包含 `PY` 的目录和 `DeepKE` 目录放入 `.env` 中：

```
DEEPKE_PATH="~/DeepKE"
CONDA_PY="/home/user_name/anaconda3/envs/deepke/bin/"
CONDA_EE_PY="/home/user_name/anaconda3/envs/deepke-ee/bin/"
```

### muKG 配置

可以完全按照[官方仓库](https://github.com/nju-websoft/muKG)来配置虚拟环境，此下重复一遍。

例如：
```bash
# command for PyTorch
conda create -n muKG python=3.8
conda activate muKG
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge python-igraph
pip install -U ray==1.12.0

git clone https://github.com/nju-websoft/muKG.git muKG
cd muKG
pip install -e .
```

**写入 `.env` 文件**

```
MUKG_PATH="~/muKG"
CONDA_MUKG_PY="/home/user_name/anaconda3/envs/muKG/bin/"
MUKG_OUTPUT_DIR="~/output/"
```

### API_KEY

使用了阿里的qwen大模型，更改 `DASHSCOPE_API_KEY` 即可，用于演示。如果只需要将 mcp 服务嵌入到应用程序（如 Cursor ）则不需要。

### 配置 MCP 项目 uv 环境

```bash
# curl -LsSf https://astral.sh/uv/install.sh | sh # 安装uv
# pipx install uv # 反正选一个安装uv
pip install uv

cd OpenKG-ToolAgent
uv venv
source .venv/bin/activate
uv add "mcp[cli]" httpx openai pyyaml
```

### 测试运行

相当于演示 demo ，也可以在这里直接对话。

```bash
cd OpenKG-ToolAgent
uv run run.py
```

### 在 Cursor & Cline 中使用

修改 `.cursor/mcp.json` 或 `cline_mcp_settings.json` ，或者在其他支持 mcp 服务的应用中添加类似的配置。

```json
{
  "mcpServers": {
    "OpenKG-ToolAgent": {
      "command": "uv",
      "args": [
        "--directory",
        "absolute/path/to/OpenKG-ToolAgent/tools",
        "run",
        "server.py"
      ]
    }
  }
}
```

## 3. 支持的能力

`OpenKG-ToolAgent` 当前提供以下任务接口：

| 任务类型 | 接口名           | 功能概述                              |
| -------- | ---------------- | ------------------------------------- |
| NER      | `deepke_ner()`   | 常规全监督命名实体识别预测            |
| RE       | `deepke_re()`    | 常规全监督关系抽取预测                |
| AE       | `deepke_ae()`    | 常规全监督属性抽取预测                |
| EE       | `deepke_ee()`    | 常规全监督事件检测及论元提取预测      |
| EA       | `ea_modelname()` | 实体对齐，包含 `MTransE` 等四个模型   |
| LP       | `lp_modelname()` | 链路预测，包含 `TransE` 等四个模型    |
| ET       | `et_transe_et()` | 实体类型识别，以 `TransE_ET` 模型运行 |

## 4. 注意事项

- 该 MCP 服务目前仍然处于早期阶段，相关能力正在完善中。
- 当前默认适配 Linux 路径，Windows 需修改对应路径。