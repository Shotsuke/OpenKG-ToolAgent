# OpenKG-ToolAgent

------

## 1. 项目简介

从“工具调用”到“任务自动化”

在真实应用中，用户往往只提出**目标级需求**，例如：

> “从这些文本中抽取实体关系”
>  “我想得到结构化的知识，而不关心中间步骤”

但现有系统通常要求用户显式指定 **NER → RE → 格式转换** 等操作流程，或者需要手动操作。这在复杂任务中既低效，也不具备可扩展性。

------

### 核心思想

一方面，**OpenKG-ToolAgent** 提供了若干工具的 **MCP** 服务；

另一方面，我们正在将 **OpenKG-ToolAgent** 打造位一个以 **Agent 调度**为核心的知识图谱工具协同框架，旨在将一个用户目标视为**多阶段、具备依赖关系的任务**，交由 Agent 自动规划与执行所需的工具调用。

------

## 2. MCP 工具

OpenKG-ToolAgent 通过 MCP（Model Context Protocol）将各类知识图谱工具封装为 **可被 Agent 调度的能力单元（Tool）**。

### DeepKE（知识抽取）

支持以下标准 NLP 抽取任务：

- 命名实体识别（NER）
- 关系抽取（RE）
- 属性抽取（AE）
- 事件抽取（EE）

本项目为其提供统一的 MCP 服务接口，支持预测阶段的独立调用。

------

### muKG（知识表示学习）

支持多源或单源知识图谱的表示学习任务：

- 实体对齐（EA）
- 链路预测（LP）
- 实体类型识别（ET）

同样以 MCP 服务形式集成，支持训练与预测。

------

### Medical_Guideline_Extract（领域扩展示例）

用于医疗指南文本的结构化抽取，展示 OpenKG-ToolAgent 在垂直领域中的可扩展性。

------

## 3. 应用场景：自动知识结构化

给定一份原始文本（如科研文献、领域说明、医疗指南），用户希望直接获得结构化的知识结果，而不需要关心中间的 NLP 流程。

**示例输入：**

```
文本文件（每行一个句子）
```

**用户目标：**

对这些文本执行操作（如关系抽取 RE）

------

### Agent 自动化行为

在 OpenKG-ToolAgent 中，系统将该需求解释为一个**具备隐式依赖的任务**：

- 关系抽取（RE）依赖于实体识别（NER）
- 当前输入中未显式提供实体信息
- 因此需要自动补全缺失步骤

------

## 4. 使用与部署

### **DeepKE 配置**

请参考 [DeepKE官方仓库](https://github.com/zjunlp/DeepKE)，配置deepke / deepke-ee虚拟环境

<!-- ```bash
git clone https://github.com/zjunlp/DeepKE.git
```

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

配置完deepke后再来配置deepke-ee，基本是相同的，不过多加一条 `pip install hydra-core==1.3.1` ， `EE` 任务使用版本更高的 `hydra` 。 -->

**写入 `.env` 文件**

将 `conda` 对应的包含 `PY` 的目录和 `DeepKE` 目录放入 `.env` 中：

```
DEEPKE_PATH="~/DeepKE"
CONDA_PY="/home/user_name/anaconda3/envs/deepke/bin/"
CONDA_EE_PY="/home/user_name/anaconda3/envs/deepke-ee/bin/"
```

### muKG 配置

可以完全按照[官方仓库](https://github.com/nju-websoft/muKG)来配置虚拟环境。

<!-- 例如：
```bash
# command for PyTorch
conda create -n muKG python=3.8
conda activate muKG
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge python-igraph
pip install -r requirements.txt
pip install -U ray==1.12.0

git clone https://github.com/nju-websoft/muKG.git muKG
cd muKG
pip install -e .
``` -->

**写入 `.env` 文件**

```
MUKG_PATH="~/muKG"
CONDA_MUKG_PY="/home/user_name/anaconda3/envs/muKG/bin/"
MUKG_OUTPUT_DIR="~/output/"
```
### MGE配置
请按照[官方仓库](https://github.com/Dustzm/Medical_Guideline_Extract.git)README要求配置环境参数。

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
uv add "mcp[cli]" httpx openai pyyaml langchain langgraph langchain-community langgraph-cli langchain-mcp-adapters ddgs requests
```

### 作为服务器

使用 `uvicorn server_api:app --reload --host 0.0.0.0 --port 8000` 来运行服务器。

### 在 Cursor & Cline 中使用

修改 `.cursor/mcp.json` 或 `cline_mcp_settings.json` ，或者在其他支持 mcp 服务的应用中添加类似的配置。

```json

{
 "mcpServers": {
  "OpenKG-ToolAgent": {
   "command": "uv",
   "args": [
​    "--directory",
​    "absolute/path/to/OpenKG-ToolAgent/tools",
​    "run",
​    "server.py"
   ]
  }
 }
}
```

------

## 3. 支持的能力

`OpenKG-ToolAgent` 的 MCP 部分当前提供以下任务接口：

| 任务类型 | 接口名           | 功能概述                              |
| -------- | ---------------- | ------------------------------------- |
| NER      | `deepke_ner()`   | 常规全监督命名实体识别预测            |
| RE       | `deepke_re()`    | 常规全监督关系抽取预测                |
| AE       | `deepke_ae()`    | 常规全监督属性抽取预测                |
| EE       | `deepke_ee()`    | 常规全监督事件检测及论元提取预测      |
| EA       | `ea_modelname()` | 实体对齐，包含 `MTransE` 等四个模型   |
| LP       | `lp_modelname()` | 链路预测，包含 `TransE` 等四个模型    |
| ET       | `et_transe_et()` | 实体类型识别，以 `TransE_ET` 模型运行 |
| MGE      | `mge_judge()`    | 医疗指南文本判断                      |
| MGE      | `mge_extract()`  | 医疗指南结构化抽取                    |

## 5. Demo 示例

```
cd OpenKG-ToolAgent
uv run run.py
```

该 Demo 演示了一个典型的 Agent 调度流程：

- 用户提出高层目标
- Agent 自动规划所需工具链
- 系统输出最终结构化结果

------

## 6. 项目状态与说明

- 本项目当前仍处于早期阶段
- Agent 调度策略以规则为主，后续将逐步模块化与可学习化
- 当前默认适配 Linux 环境
