"""muKG 模型统一运行接口模块。

该模块封装了 muKG 框架中不同任务的统一调用入口。

环境变量要求：
    - MUKG_PATH: muKG 项目根路径。
    - CONDA_MUKG_PY: 运行 muKG 脚本的 Python 环境调用命令。
    - MUKG_OUTPUT_DIR: 运行 muKG 得到的输出存放位置。

该模块提供以下函数：
    - run_mukg_command(): 统一执行命令。
    - mukg_ea(): 实体识别。
    - mukg_lp(): 链接预测。
    - mukg_et(): 实体类型。
    - check_task(): 查询任务。
"""

import subprocess
import os
import uuid
import time
from typing import Dict, Optional, List
from datetime import datetime
import threading
import glob

# 正在运行的任务与对应的日志文件映射
running_tasks: Dict[str, subprocess.Popen] = {}
task_log_files: Dict[str, str] = {}


def run_mukg_command(task_type: str, model: str, train: bool, data: str) -> str:
    """
    通用的 muKG 命令运行函数。

    根据任务类型、模型名称和训练模式生成命令行，并在后台以异步方式运行。
    任务执行日志会输出到指定目录，并可通过任务 ID 查询状态。

    Args:
        task_type (str): 任务类型，如 "ea"（实体对齐）、"lp"（链接预测）、"et"（实体类型）。
        model (str): 所使用的模型名称，例如 "MTransE"、"gcnalign"、"tucker" 等。
        train (bool): 若为 True，则执行训练与测试；若为 False，则仅执行测试。
        data (str): 数据集路径或名称（相对 muKG 数据目录）。

    Returns:
        str: 启动状态信息或错误提示字符串。
    """
    base_path = os.path.expanduser(os.path.join(
        os.getenv("MUKG_PATH"), "src/py"
    ))
    task_id = str(uuid.uuid4())
    log_filename = f"mukg_{'train' if train else 'test'}_{model}_{task_id}.log"
    log_file = os.path.expanduser(os.path.join(
        os.getenv("MUKG_OUTPUT_DIR"), log_filename
    ))
    train_cmd = "train" if train else "test"

    command = [
        os.getenv('CONDA_MUKG_PY') + "python",
        "main_args.py",
        "-t", task_type,
        "-m", model,
        "-o", train_cmd,
        "-d", "data/" + data
    ]

    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        with open(log_file, "w") as f:
            process = subprocess.Popen(
                command,
                cwd=base_path,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        # 记录任务信息
        running_tasks[task_id] = process
        task_log_files[task_id] = log_file

        # 启动线程监控任务完成状态
        def monitor_task(proc, task_id):
            proc.wait()
            if task_id in running_tasks:
                del running_tasks[task_id]

        threading.Thread(target=monitor_task, args=(process, task_id), daemon=True).start()

        return f"进程已启动，任务编号 {task_id} ，请保存该编号以查询任务状态。"

    except Exception as e:
        return f"[运行失败] {str(e)}"


def mukg_ea(model: str, train: bool, data: str) -> str:
    """
    使用 muKG 运行实体对齐（Entity Alignment, EA）任务。

    Args:
        model (str): 模型名称，例如 "MTransE"、"gcnalign"、"bootea" 等。
        train (bool): 若为 True，则执行训练与测试；若为 False，则仅执行测试。
        data (str): 数据集路径或名称。

    Returns:
        str: 任务启动信息或错误提示。
    """
    return run_mukg_command("ea", model, train, data)


def mukg_lp(model: str, train: bool, data: str) -> str:
    """
    使用 muKG 运行链接预测（Link Prediction, LP）任务。

    Args:
        model (str): 模型名称，例如 "transe"、"rotate"、"tucker" 等。
        train (bool): 若为 True，则执行训练与测试；若为 False，则仅执行测试。
        data (str): 数据集名称，例如 "FB15K"。

    Returns:
        str: 任务启动信息或错误提示。
    """
    return run_mukg_command("lp", model, train, data)


def mukg_et(model: str, train: bool, data: str) -> str:
    """
    使用 muKG 运行实体类型预测（Entity Typing, ET）任务。

    Args:
        model (str): 模型名称，例如 "transe"。
        train (bool): 若为 True，则执行训练与测试；若为 False，则仅执行测试。
        data (str): 数据集名称，例如 "FB15K_type"。

    Returns:
        str: 任务启动信息或错误提示。
    """
    return run_mukg_command("et", model, train, data)


def check_task(task_id: str) -> str:
    """
    检查指定任务的当前状态或查看其日志内容。

    若任务仍在运行，则返回“进行中”；若任务已完成，则返回日志的前后片段。

    Args:
        task_id (str): 用于查询的任务唯一编号（UUID）。

    Returns:
        str: 任务状态、日志摘要或错误信息。
    """
    # 检查是否是运行中的任务
    if task_id in running_tasks:
        return f"任务 {task_id} 正在进行中..."

    # 查找日志文件
    log_pattern = os.path.expanduser(os.path.join(
        os.getenv("MUKG_OUTPUT_DIR"), f"*{task_id}*.log"
    ))
    log_files = glob.glob(log_pattern)

    if not log_files:
        return f"未找到任务 {task_id}，请确认任务编号是否正确。"

    # 读取日志内容
    log_file = log_files[0]
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        if not lines:
            return f"任务 {task_id} 已完成，但日志文件为空。"

        head_lines = lines[:10]
        tail_lines = lines[-10:] if len(lines) > 20 else lines

        result = f"任务 {task_id} 已完成。\n\n日志头部:\n"
        result += "".join(head_lines)
        result += "\n\n日志尾部:\n"
        result += "".join(tail_lines)

        return result

    except Exception as e:
        return f"读取任务 {task_id} 日志时出错: {str(e)}"
