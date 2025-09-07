import subprocess
import os
import uuid
import time
from typing import Dict, Optional, List
from datetime import datetime
import threading
import glob

running_tasks: Dict[str, subprocess.Popen] = {}
task_log_files: Dict[str, str] = {}

def run_mukg_command(task_type: str, model: str, train: bool, data: str) -> str:
    """通用的muKG命令运行函数"""

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

        # 存储任务信息
        running_tasks[task_id] = process
        task_log_files[task_id] = log_file
        
        # 启动线程监控任务完成
        def monitor_task(proc, task_id):
            proc.wait()
            if task_id in running_tasks:
                del running_tasks[task_id]
        
        threading.Thread(target=monitor_task, args=(process, task_id), daemon=True).start()

        return f"进程已启动，任务编号 {task_id} ，保存编号以查询任务。"

    except Exception as e:
        return f"[运行失败] {str(e)}"

def mukg_ea(model: str, train: bool, data: str) -> str:
    """
    使用 muKG 运行实体对其 ea 任务 
    """
    return run_mukg_command("ea", model, train, data)

def mukg_lp(model: str, train: bool, data: str) -> str:
    """
    使用 muKG 运行链接预测 lp 任务 
    """
    return run_mukg_command("lp", model, train, data)

def mukg_et(model: str, train: bool, data: str) -> str:
    """
    使用 muKG 运行实体类型 et 任务 
    """
    return run_mukg_command("et", model, train, data)

def check_task(task_id: str) -> str:
    """检查任务状态"""
    # 检查是否是运行中的任务
    if task_id in running_tasks:
        return f"任务 {task_id} 正在进行中..."
    
    # 检查是否有对应的日志文件
    log_pattern = os.path.expanduser(os.path.join(
        os.getenv("MUKG_OUTPUT_DIR"), f"*{task_id}*.log"
    ))
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        return f"未找到任务 {task_id}，请确认任务编号是否正确。"
    
    # 读取日志文件的首尾部分
    log_file = log_files[0]
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return f"任务 {task_id} 已完成，但日志文件为空。"
        
        # 获取头部和尾部各10行
        head_lines = lines[:10]
        tail_lines = lines[-10:] if len(lines) > 20 else lines
        
        result = f"任务 {task_id} 已完成。\n\n日志头部:\n"
        result += "".join(head_lines)
        result += "\n\n日志尾部:\n"
        result += "".join(tail_lines)
        
        return result
    except Exception as e:
        return f"读取任务 {task_id} 日志时出错: {str(e)}"