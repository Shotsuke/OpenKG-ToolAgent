"""Medical Guideline Extract服务集成
该模块封装了 Medical Guideline Extract 框架中的不同任务。

环境变量要求：
    - MGEX_PATH: Medical Guideline Extract 项目根路径。
    - CONDA_MG_EXTRACT_PY: 运行 Medical Guideline Extract 脚本的 Python 环境调用命令。

该模块提供以下函数：
    - judge_content(): 医疗指南内容判别。
    - knowledge_extract(): 医疗指南内容结构化抽取。
"""
import os
import subprocess
import yaml
from typing import Literal



def mgex_judge(content: str) -> str:
    """
    文本内容类型判断，判断文本是否为医疗指南内容
    Args:
        content: 文本内容

    Returns:
        bool: 是否为医疗指南内容

    """
    base_path = os.path.expanduser(os.path.join(
        os.getenv("MGEX_PATH"), "backend/mcp_support"
    ))
    content = content.replace('\n',' ').strip()

    try:
        # 执行预测脚本
        command = f"cd {base_path} && {os.getenv('CONDA_MG_EXTRACT_PY')}python judge.py {content}"
        result = subprocess.run(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout
    except Exception as e:
        return f"[运行失败] {str(e)}"

def mgex_extract(content: str) -> str:
    """
    医疗指南内容结构化抽取
    Args:
        content: 文本内容

    Returns:
        str: 结构化抽取结果

    """
    base_path = os.path.expanduser(os.path.join(
        os.getenv("MGEX_PATH"), "backend/mcp_support"
    ))
    content = content.replace('\n', ' ').strip()

    try:
        # 执行预测脚本
        command = f"cd {base_path} && {os.getenv('CONDA_MG_EXTRACT_PY')}python extract.py {content}"
        result = subprocess.run(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout
    except Exception as e:
        return f"[运行失败] {str(e)}"
