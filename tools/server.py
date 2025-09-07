import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from deepke import deepke_ner, deepke_re, deepke_ae, deepke_ee
from mukg import mukg_ea, mukg_lp, mukg_et, check_task

load_dotenv()

mcp = FastMCP("OpenKG")

# 注册工具
@mcp.tool()
def ner(task: str, txt: str) -> str:
    """
    使用 deepke 运行 NER 预测任务
    
    参数:
    - task: 任务类型（目前只支持standard）
    - txt: 用户的预测文本
    
    返回:
    - 命令输出或错误信息
    """
    return deepke_ner(task, txt)

@mcp.tool()
def re(task: str, txt: str, head: str, head_type: str, tail: str, tail_type: str) -> str:
    """
    使用 deepke 运行关系抽取(RE)预测任务
    
    参数:
    - task: 任务类型（目前只支持standard）
    - txt: 包含实体的句子
    - head: 句中需要预测关系的头实体
    - head_type: 头实体类型
    - tail: 句中需要预测关系的尾实体
    - tail_type: 尾实体类型
    
    返回:
    - 命令输出或错误信息
    """
    return deepke_re(task, txt, head, head_type, tail, tail_type)

@mcp.tool()
def ae(txt: str, entity: str, attribute_value: str, task: str = 'standard') -> str:
    """
    使用 deepke 运行属性抽取(AE)预测任务
    
    参数:
    - txt: 需要做属性抽取的句子
    - entity: 句中需要预测的实体
    - attribute_value: 句中需要预测的属性值
    
    返回:
    - 命令输出或错误信息
    """
    return deepke_ae(txt, entity, attribute_value, task)

@mcp.tool()
def ee(txt: str) -> str:
    """
    使用 deepke 运行事件抽取 EE 预测任务
    
    参数:
    - txt: 用户的预测文本
    
    返回:
    - 命令输出或错误信息
    """
    return deepke_ee(txt)

@mcp.tool()
def ea_mtranse(train: bool, data: str = "OpenEA_dataset_v1.1/EN_FR_15K_V1/") -> str:
    """
    使用 muKG 库中整合的 MTransE 模型来运行实体对齐 ea 任务 

    参数:
    - train: 设为 True 时执行训练 + 测试，设为 False 时跳过训练
    - data: 数据集选择(默认 OpenEA_dataset_v1.1/EN_FR_15K_V1/ )

    返回:
    - 命令输出或错误信息
    """
    
    if not data or data.strip() == "":
        data = "OpenEA_dataset_v1.1/EN_FR_15K_V1/"

    return mukg_ea("MTransE", train, data)

@mcp.tool()
def ea_gcnalign(train: bool, data: str = "OpenEA_dataset_v1.1/EN_FR_15K_V1/") -> str:
    """
    使用 muKG 库中整合的 GCN-Align 模型来运行实体对齐 ea 任务 

    参数:
    - train: 设为 True 时执行训练 + 测试，设为 False 时跳过训练
    - data: 数据集选择(默认 OpenEA_dataset_v1.1/EN_FR_15K_V1/)

    返回:
    - 命令输出或错误信息
    """
    
    if not data or data.strip() == "":
        data = "OpenEA_dataset_v1.1/EN_FR_15K_V1/"

    return mukg_ea("gcnalign", train, data)

@mcp.tool()
def ea_bootea(train: bool, data: str = "OpenEA_dataset_v1.1/EN_FR_15K_V1/") -> str:
    """
    使用 muKG 库中整合的 BootEA 模型来运行实体对齐 ea 任务 

    参数:
    - train: 设为 True 时执行训练 + 测试，设为 False 时跳过训练
    - data: 数据集选择(默认 OpenEA_dataset_v1.1/EN_FR_15K_V1/ )

    返回:
    - 命令输出或错误信息
    """
    
    if not data or data.strip() == "":
        data = "OpenEA_dataset_v1.1/EN_FR_15K_V1/ "

    return mukg_ea("bootea", train, data)

@mcp.tool()
def ea_rsn4ea(train: bool, data: str = "OpenEA_dataset_v1.1/EN_FR_15K_V1/") -> str:
    """
    使用 muKG 库中整合的 RSN4EA 模型来运行实体对齐 ea 任务 

    参数:
    - train: 设为 True 时执行训练 + 测试，设为 False 时跳过训练
    - data: 数据集选择(默认 OpenEA_dataset_v1.1/EN_FR_15K_V1/ )

    返回:
    - 命令输出或错误信息
    """
    
    if not data or data.strip() == "":
        data = "OpenEA_dataset_v1.1/EN_FR_15K_V1/ "

    return mukg_ea("rsn4ea", train, data)

@mcp.tool()
def lp_transe(train: bool, data: str = "FB15K") -> str:
    """
    使用 muKG 库中整合的 TransE 模型运行链接预测 lp 任务 

    参数:
    - train: 设为 True 时执行训练 + 测试，设为 False 时跳过训练
    - data: 数据集选择(默认 FB15K )

    返回:
    - 命令输出或错误信息
    """
    
    if not data or data.strip() == "":
        data = "FB15K"   

    return mukg_lp("transe", train, data)

@mcp.tool()
def lp_rotate(train: bool, data: str = "FB15K") -> str:
    """
    使用 muKG 库中整合的 RotatE 模型运行链接预测 lp 任务 

    参数:
    - train: 设为 True 时执行训练 + 测试，设为 False 时跳过训练
    - data: 数据集选择(默认 FB15K )

    返回:
    - 命令输出或错误信息
    """
    
    if not data or data.strip() == "":
        data = "FB15K"   

    return mukg_lp("rotate", train, data)

@mcp.tool()
def lp_conve(train: bool, data: str = "FB15K") -> str:
    """
    使用 muKG 库中整合的 ConvE 模型运行链接预测 lp 任务 

    参数:
    - train: 设为 True 时执行训练 + 测试，设为 False 时跳过训练
    - data: 数据集选择(默认 FB15K )

    返回:
    - 命令输出或错误信息
    """
    
    if not data or data.strip() == "":
        data = "FB15K"   

    return mukg_lp("conve", train, data)

@mcp.tool()
def lp_tucker(train: bool, data: str = "FB15K") -> str:
    """
    使用 muKG 库中整合的 TuckER 模型运行链接预测 lp 任务 

    参数:
    - train: 设为 True 时执行训练 + 测试，设为 False 时跳过训练
    - data: 数据集选择(默认 FB15K )

    返回:
    - 命令输出或错误信息
    """
    
    if not data or data.strip() == "":
        data = "FB15K"   

    return mukg_lp("tucker", train, data)

@mcp.tool()
def et_transe_et(train: bool, data: str = "FB15K_type") -> str:
    """
    使用 muKG 库中整合的 TransE_ET 模型运行实体类型 et 任务 

    参数:
    - train: 设为 True 时执行训练 + 测试，设为 False 时跳过训练
    - data: 数据集选择(默认 FB15K_type )

    返回:
    - 命令输出或错误信息
    """
    
    if not data or data.strip() == "":
        data = "FB15K_type"   

    return mukg_et("transe", train, data)

@mcp.tool()
def mukg_check(task_id: str) -> str:
    """
    检查使用 muKG 进行训练、测试的模型是否完成

    参数：
    - task_id: ID 值用于查询任务。
    """
    return check_task(task_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
