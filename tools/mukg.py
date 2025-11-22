"""muKG 模型统一运行接口模块。

该模块封装了 muKG 框架中不同任务的统一调用入口。

环境变量要求：
    - MUKG_PATH: muKG 项目根路径。
    - CONDA_MUKG_PY: 运行 muKG 脚本的 Python 环境调用命令。
    - MUKG_LOG_OUTPUT_DIR: 运行 muKG 得到的输出存放位置。

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
import shutil
import zipfile

# 正在运行的任务与对应的日志文件映射
running_tasks: Dict[str, subprocess.Popen] = {}
task_log_files: Dict[str, str] = {}

# 模型名称映射表：将外部调用名称映射到实际的目录名称
MODEL_NAME_MAPPING = {
    # EA 任务
    "mtranse": "MTransE",
    "mtransE": "MTransE",
    "gcn_align": "GCN_Align", 
    "gcnalign": "GCN_Align",
    "bootea": "BootEA",
    "rrea": "RREA",
    "nmea": "NMEA",
    "attr2vec": "Attr2Vec",
    "imuse": "IMUSE",
    "kegcn": "KEGCN",
    "multi_ke": "MultiKE",
    
    # LP 任务
    "transe": "TransE",
    "transh": "TransH",
    "transr": "TransR",
    "transd": "TransD",
    "distmult": "DistMult",
    "complex": "ComplEx",
    "rotate": "RotatE",
    "tucker": "TuckER",
    "conve": "ConvE",
    "convkb": "ConvKB",
    
    # ET 任务
    "transe_et": "TransE",
    "transh_et": "TransH",
    "transr_et": "TransR",
    "transd_et": "TransD",
}


def get_actual_model_dir(mukg_root: str, model_name: str) -> str:
    """
    根据模型名称映射表获取实际的模型目录路径。
    
    Args:
        mukg_root: muKG 项目根路径
        model_name: 模型名称
    
    Returns:
        str: 实际的模型目录路径
    """
    base_dir = os.path.join(mukg_root, "output", "results")
    
    # 首先尝试映射表
    mapped_name = MODEL_NAME_MAPPING.get(model_name.lower())
    if mapped_name:
        mapped_path = os.path.join(base_dir, mapped_name)
        if os.path.exists(mapped_path):
            return mapped_path
    
    # 如果映射表没有找到，尝试原始名称
    original_path = os.path.join(base_dir, model_name)
    if os.path.exists(original_path):
        return original_path
    
    # 如果原始名称也不存在，尝试一些常见的变体
    possible_variants = [
        model_name.capitalize(),
        model_name.upper(),
        model_name.lower(),
        model_name.replace('_', ''),
        model_name.replace('_', '-'),
    ]
    
    for variant in possible_variants:
        variant_path = os.path.join(base_dir, variant)
        if os.path.exists(variant_path):
            return variant_path
    
    # 最后尝试在目录中查找包含模型名称的文件夹
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            if model_name.lower() in item.lower():
                found_path = os.path.join(base_dir, item)
                if os.path.isdir(found_path):
                    return found_path
    
    # 如果都找不到，返回基于映射名称的路径（即使不存在）
    return os.path.join(base_dir, mapped_name or model_name)


def cleanup_model_files(model_dir: str, task_id: str, log_file: str):
    """
    清理模型文件以释放服务器空间。
    
    Args:
        model_dir: 模型目录路径
        task_id: 任务ID
        log_file: 日志文件路径
    """
    try:
        if os.path.exists(model_dir):
            # 计算删除前的目录大小
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(model_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            # 删除模型目录
            shutil.rmtree(model_dir)
            
            # 记录清理信息
            with open(log_file, "a") as f:
                f.write(f"\n[INFO] 已清理模型文件: {model_dir}\n")
                f.write(f"[INFO] 释放空间: {total_size / (1024**3):.2f} GB\n")
                f.write(f"[INFO] 任务 {task_id} 的模型文件已清理完成\n")
        else:
            with open(log_file, "a") as f:
                f.write(f"\n[INFO] 模型目录不存在，无需清理: {model_dir}\n")
                
    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"\n[ERROR] 清理模型文件失败: {str(e)}\n")
            import traceback
            f.write(f"[ERROR] 详细错误: {traceback.format_exc()}\n")


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
        os.getenv("MUKG_LOG_OUTPUT_DIR"), log_filename
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
        def monitor_task(proc, task_id, model_name, is_training):
            """
            监控任务完成状态，并在训练完成后自动打包模型文件。
            
            Args:
                proc: 子进程对象
                task_id: 任务ID
                model_name: 模型名称
                is_training: 是否为训练模式
            """
            proc.wait()
            if task_id in running_tasks:
                del running_tasks[task_id]
            
            # 等待输出稳定
            time.sleep(5)
            
            # 只有在训练模式下才打包模型文件
            if is_training:
                try:
                    # === 自动打包模型文件 ===
                    mukg_root = os.path.expanduser(os.getenv("MUKG_PATH") or "~/muKG")
                    
                    # 使用映射表获取实际的模型目录
                    model_out_dir = get_actual_model_dir(mukg_root, model_name)
                    
                    # 检查输出目录是否存在
                    if not os.path.exists(model_out_dir):
                        with open(task_log_files[task_id], "a") as f:
                            f.write(f"\n[INFO] 模型输出目录不存在: {model_out_dir}\n")
                            f.write(f"[INFO] 尝试查找包含 '{model_name}' 的目录...\n")
                        return
                    
                    # 寻找最新的输出子目录
                    latest_dir = None
                    latest_time = 0
                    
                    for root, dirs, files in os.walk(model_out_dir):
                        for dir_name in dirs:
                            dir_path = os.path.join(root, dir_name)
                            dir_time = os.path.getmtime(dir_path)
                            if dir_time > latest_time:
                                latest_time = dir_time
                                latest_dir = dir_path
                    
                    if not latest_dir:
                        with open(task_log_files[task_id], "a") as f:
                            f.write(f"\n[INFO] 在 {model_out_dir} 中未找到模型输出子目录\n")
                        return
                    
                    # 创建服务器静态文件目录（用于下载）
                    server_static_root = os.path.expanduser("~/OpenKG-ToolAgent/output/models")
                    task_dir = os.path.join(server_static_root, task_id)
                    os.makedirs(task_dir, exist_ok=True)
                    
                    # 创建zip文件
                    zip_filename = f"{model_name}_{task_id}.zip"
                    zip_path = os.path.join(task_dir, zip_filename)
                    
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        file_count = 0
                        for root, _, files in os.walk(latest_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                # 在zip文件中保持相对路径结构
                                arcname = os.path.relpath(file_path, start=os.path.dirname(latest_dir))
                                zf.write(file_path, arcname)
                                file_count += 1
                    
                    # 写入成功日志
                    with open(task_log_files[task_id], "a") as f:
                        f.write(f"\n[INFO] 模型已成功打包: {zip_path}\n")
                        f.write(f"[INFO] 模型文件数量: {file_count}\n")
                        f.write(f"[INFO] 打包源目录: {latest_dir}\n")
                        f.write(f"[INFO] 模型映射: {model_name} -> {os.path.basename(model_out_dir)}\n")
                        f.write(f"[INFO] 下载链接: /output/models/{task_id}/{zip_filename}\n")
                    
                    # === 清理原始模型文件以释放空间 ===
                    cleanup_model_files(latest_dir, task_id, task_log_files[task_id])
                        
                except Exception as e:
                    # 记录打包失败信息
                    with open(task_log_files[task_id], "a") as f:
                        f.write(f"\n[ERROR] 模型打包失败: {str(e)}\n")
                        import traceback
                        f.write(f"[ERROR] 详细错误: {traceback.format_exc()}\n")

        # 启动监控线程
        threading.Thread(
            target=monitor_task, 
            args=(process, task_id, model, train),
            daemon=True
        ).start()

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
    查询任务状态和结果。

    Args:
        task_id (str): 任务ID。

    Returns:
        str: 任务状态信息和结果摘要。
    """
    # 检查任务是否仍在运行
    if task_id in running_tasks:
        return f"任务 {task_id} 正在进行中..."

    # 查找对应的日志文件
    log_pattern = os.path.expanduser(os.path.join(
        os.getenv("MUKG_LOG_OUTPUT_DIR"), f"*{task_id}*.log"
    ))
    log_files = glob.glob(log_pattern)
    if not log_files:
        return f"未找到任务 {task_id}，请确认任务编号是否正确。"

    log_file = log_files[0]
    
    # 读取日志内容
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return f"读取日志文件失败: {str(e)}"

    # 获取日志头部和尾部
    head = "".join(lines[:10]) if len(lines) >= 10 else "".join(lines)
    tail = "".join(lines[-20:]) if len(lines) > 20 else "".join(lines[-5:]) if len(lines) > 5 else ""

    # 检查是否有打包的模型文件（现在在服务器静态目录）
    server_static_root = os.path.expanduser("~/OpenKG-ToolAgent/output/models")
    task_dir = os.path.join(server_static_root, task_id)
    model_files = []
    if os.path.exists(task_dir):
        model_files = glob.glob(os.path.join(task_dir, "*.zip"))

    # 构建返回结果
    result = f"任务 {task_id} 已完成。\n\n日志头部:\n{head}\n\n日志尾部:\n{tail}"
    
    if model_files:
        links = "\n".join([f"http://ws.nju.edu.cn/toolagent/output/models/{task_id}/{os.path.basename(f)}" for f in model_files])
        result += f"\n\n模型文件可下载：\n{links}"
        result += f"\n\n注意：原始模型文件已自动清理以释放服务器空间，请及时下载保存。"
    else:
        result += "\n\n未检测到打包模型文件（可能因为这是测试任务或打包失败）。"

    return result


def list_available_models(task_type: str = "all") -> Dict[str, List[str]]:
    """
    列出可用的模型名称。
    
    Args:
        task_type: 任务类型，"ea", "lp", "et", 或 "all"
    
    Returns:
        模型名称字典
    """
    models = {
        "ea": ["MTransE", "GCN_Align", "BootEA", "RREA", "NMEA", "Attr2Vec", "IMUSE", "KEGCN", "MultiKE"],
        "lp": ["TransE", "TransH", "TransR", "TransD", "DistMult", "ComplEx", "RotatE", "TuckER", "ConvE", "ConvKB"],
        "et": ["TransE", "TransH", "TransR", "TransD"]
    }
    
    if task_type == "all":
        return models
    elif task_type in models:
        return {task_type: models[task_type]}
    else:
        return {"error": f"未知的任务类型: {task_type}"}