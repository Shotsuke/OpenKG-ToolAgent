"""
OpenKG FastMCP 工具集

本模块通过 FastMCP 注册 DeepKE 与 muKG 的一系列知识图谱任务工具，
包括以下任务：
- NER: 命名实体识别（Named Entity Recognition）
- RE: 关系抽取（Relation Extraction）
- AE: 属性抽取（Attribute Extraction）
- EE: 事件抽取（Event Extraction）
- EA: 实体对齐（Entity Alignment）
- LP: 链接预测（Link Prediction）
- ET: 实体类型预测（Entity Typing）

每个工具函数均返回模型的运行结果或错误信息字符串。
"""

import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from deepke import deepke_ner, deepke_re, deepke_ae, deepke_ee
from mukg import mukg_ea, mukg_lp, mukg_et, check_task
from mge import mgex_judge, mgex_extract

load_dotenv()

mcp = FastMCP("OpenKG")


@mcp.tool()
def ner(task: str, txt: str) -> str:
    """
    使用 DeepKE 运行命名实体识别（NER）预测任务。

    Args:
        task (str): 任务类型（目前仅支持 'standard'）。
        txt (str): 用户输入的预测文本。

    Returns:
        str: 模型输出结果或错误信息。
    """
    return deepke_ner(task, txt)


@mcp.tool()
def re(task: str, txt: str, head: str, head_type: str, tail: str, tail_type: str) -> str:
    """
    使用 DeepKE 运行关系抽取（RE）预测任务。

    Args:
        task (str): 任务类型（目前仅支持 'standard'）。
        txt (str): 包含实体的句子。
        head (str): 句中需要预测关系的头实体。
        head_type (str): 头实体类型。
        tail (str): 句中需要预测关系的尾实体。
        tail_type (str): 尾实体类型。

    Returns:
        str: 模型输出结果或错误信息。
    """
    return deepke_re(task, txt, head, head_type, tail, tail_type)


@mcp.tool()
def ae(txt: str, entity: str, attribute_value: str, task: str = 'standard') -> str:
    """
    使用 DeepKE 运行属性抽取（AE）预测任务。

    Args:
        txt (str): 需要进行属性抽取的句子。
        entity (str): 句中需要预测属性的实体。
        attribute_value (str): 句中需要预测的属性值。
        task (str, optional): 任务类型，默认为 'standard'。

    Returns:
        str: 模型输出结果或错误信息。
    """
    return deepke_ae(txt, entity, attribute_value, task)


@mcp.tool()
def ee(txt: str) -> str:
    """
    使用 DeepKE 运行事件抽取（EE）预测任务。

    Args:
        txt (str): 用户输入的预测文本。

    Returns:
        str: 模型输出结果或错误信息。
    """
    return deepke_ee(txt)


@mcp.tool()
def ea_mtranse(train: bool, data: str = "OpenEA_dataset_v1.1/EN_FR_15K_V1/") -> str:
    """
    使用 muKG 集成的 MTransE 模型运行实体对齐（EA）任务。

    Args:
        train (bool): 若为 True，则执行训练与测试；若为 False，则跳过训练。
        data (str, optional): 数据集路径，默认为 "OpenEA_dataset_v1.1/EN_FR_15K_V1/"。

    Returns:
        str: 模型输出结果或错误信息。
    """
    if not data.strip():
        data = "OpenEA_dataset_v1.1/EN_FR_15K_V1/"
    return mukg_ea("MTransE", train, data)


@mcp.tool()
def ea_gcnalign(train: bool, data: str = "OpenEA_dataset_v1.1/EN_FR_15K_V1/") -> str:
    """
    使用 muKG 集成的 GCN-Align 模型运行实体对齐（EA）任务。

    Args:
        train (bool): 若为 True，则执行训练与测试；若为 False，则跳过训练。
        data (str, optional): 数据集路径，默认为 "OpenEA_dataset_v1.1/EN_FR_15K_V1/"。

    Returns:
        str: 模型输出结果或错误信息。
    """
    if not data.strip():
        data = "OpenEA_dataset_v1.1/EN_FR_15K_V1/"
    return mukg_ea("gcnalign", train, data)


@mcp.tool()
def ea_bootea(train: bool, data: str = "OpenEA_dataset_v1.1/EN_FR_15K_V1/") -> str:
    """
    使用 muKG 集成的 BootEA 模型运行实体对齐（EA）任务。

    Args:
        train (bool): 若为 True，则执行训练与测试；若为 False，则跳过训练。
        data (str, optional): 数据集路径，默认为 "OpenEA_dataset_v1.1/EN_FR_15K_V1/"。

    Returns:
        str: 模型输出结果或错误信息。
    """
    if not data.strip():
        data = "OpenEA_dataset_v1.1/EN_FR_15K_V1/"
    return mukg_ea("bootea", train, data)


@mcp.tool()
def ea_rsn4ea(train: bool, data: str = "OpenEA_dataset_v1.1/EN_FR_15K_V1/") -> str:
    """
    使用 muKG 集成的 RSN4EA 模型运行实体对齐（EA）任务。

    Args:
        train (bool): 若为 True，则执行训练与测试；若为 False，则跳过训练。
        data (str, optional): 数据集路径，默认为 "OpenEA_dataset_v1.1/EN_FR_15K_V1/"。

    Returns:
        str: 模型输出结果或错误信息。
    """
    if not data.strip():
        data = "OpenEA_dataset_v1.1/EN_FR_15K_V1/"
    return mukg_ea("rsn4ea", train, data)


@mcp.tool()
def lp_transe(train: bool, data: str = "FB15K") -> str:
    """
    使用 muKG 集成的 TransE 模型运行链接预测（LP）任务。

    Args:
        train (bool): 若为 True，则执行训练与测试；若为 False，则跳过训练。
        data (str, optional): 数据集名称，默认为 'FB15K'。

    Returns:
        str: 模型输出结果或错误信息。
    """
    if not data.strip():
        data = "FB15K"
    return mukg_lp("transe", train, data)


@mcp.tool()
def lp_rotate(train: bool, data: str = "FB15K") -> str:
    """
    使用 muKG 集成的 RotatE 模型运行链接预测（LP）任务。

    Args:
        train (bool): 若为 True，则执行训练与测试；若为 False，则跳过训练。
        data (str, optional): 数据集名称，默认为 'FB15K'。

    Returns:
        str: 模型输出结果或错误信息。
    """
    if not data.strip():
        data = "FB15K"
    return mukg_lp("rotate", train, data)


@mcp.tool()
def lp_conve(train: bool, data: str = "FB15K") -> str:
    """
    使用 muKG 集成的 ConvE 模型运行链接预测（LP）任务。

    Args:
        train (bool): 若为 True，则执行训练与测试；若为 False，则跳过训练。
        data (str, optional): 数据集名称，默认为 'FB15K'。

    Returns:
        str: 模型输出结果或错误信息。
    """
    if not data.strip():
        data = "FB15K"
    return mukg_lp("conve", train, data)


@mcp.tool()
def lp_tucker(train: bool, data: str = "FB15K") -> str:
    """
    使用 muKG 集成的 TuckER 模型运行链接预测（LP）任务。

    Args:
        train (bool): 若为 True，则执行训练与测试；若为 False，则跳过训练。
        data (str, optional): 数据集名称，默认为 'FB15K'。

    Returns:
        str: 模型输出结果或错误信息。
    """
    if not data.strip():
        data = "FB15K"
    return mukg_lp("tucker", train, data)


@mcp.tool()
def et_transe_et(train: bool, data: str = "FB15K_type") -> str:
    """
    使用 muKG 集成的 TransE_ET 模型运行实体类型预测（ET）任务。

    Args:
        train (bool): 若为 True，则执行训练与测试；若为 False，则跳过训练。
        data (str, optional): 数据集名称，默认为 'FB15K_type'。

    Returns:
        str: 模型输出结果或错误信息。
    """
    if not data.strip():
        data = "FB15K_type"
    return mukg_et("transe", train, data)


@mcp.tool()
def mukg_check(task_id: str) -> str:
    """
    检查使用 muKG 进行训练或测试的任务状态。

    Args:
        task_id (str): 用于查询任务的唯一 ID。

    Returns:
        str: 当前任务状态或错误信息。
    """
    return check_task(task_id)

@mcp.tool()
def mg_judge(content: str) -> str:
    """
    文本内容类型判断，判断文本是否为医疗指南内容
    Args:
        content: 文本内容，该参数内容需要使用引号包裹，例如：\\"这是一条医疗指南内容\\"

    Returns:
        str: 是否为医疗指南内容,True或False

    """
    return mgex_judge(content)

@mcp.tool()
def mg_extract(content: str, type: bool) -> str:
    """
    医疗指南内容结构化抽取，在调用本方法前应先调用judge工具对文本进行判断
    本工具需要调用大模型进行结构化抽取，耗时较长
    Args:
        content: 文本内容，该参数内容需要使用引号包裹，例如：\"这是一条医疗指南内容\"
        type: 是否为医疗指南内容

    Returns:
        str: 结构化抽取结果，csv格式

    """
    if not type:
        return "文本非医疗指南内容，无法进行结构化抽取"
    return mgex_extract(content)

if __name__ == "__main__":
    mcp.run(transport="stdio")
