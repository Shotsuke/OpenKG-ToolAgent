"""DeepKE 模型统一运行接口模块。

该模块封装了 DeepKE 框架中的不同任务。

环境变量要求：
    - DEEPKE_PATH: DeepKE 项目根路径。
    - CONDA_DEEPKE_PY: 运行 DeepKE 脚本的 Python 环境调用命令。
    - CONDA_DEEPKE_EE_PY: 运行 DeepKE 事件抽取模块的 Python 环境调用命令。

该模块提供以下函数：
    - deepke_ner(): 命名实体识别。
    - deepke_re(): 关系抽取。
    - deepke_ae(): 属性抽取。
    - deepke_ee(): 事件抽取。
"""

import os
import subprocess
import yaml
from convert_to_tsv import input_to_raw_and_tsv
from typing import Literal

TaskType = Literal["standard", "few-shot", "multimodal", "documental"]


def deepke_ner(task: TaskType, txt: str) -> str:
    """执行 DeepKE 的命名实体识别（NER）任务。

    Args:
        task (TaskType): 指定任务类型。目前仅支持 "standard"。
        txt (str): 输入文本内容。

    Returns:
        str: 模型预测输出结果字符串，或错误信息。

    Raises:
        Exception: 运行配置文件读取、写入或子进程调用失败时触发。

    Notes:
        此函数会直接修改 `example/ner/standard/conf/predict.yaml` 中的 `text` 字段。
    """
    if task != "standard":
        return "[NER任务目前只支持standard模式]"

    base_path = os.path.expanduser(os.path.join(
        os.getenv("DEEPKE_PATH"), "example/ner/standard"
    ))
    config_path = os.path.join(base_path, "conf/predict.yaml")

    try:
        # 写入预测配置
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["text"] = txt
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)

        # 执行预测脚本
        command = f"cd {base_path} && {os.getenv('CONDA_DEEPKE_PY')}python predict.py"
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


def deepke_re(task: TaskType, txt: str, head: str, head_type: str, tail: str, tail_type: str) -> str:
    """执行 DeepKE 的关系抽取（RE）任务。

    Args:
        task (TaskType): 指定任务类型。目前仅支持 "standard"。
        txt (str): 输入文本。
        head (str): 主实体名称。
        head_type (str): 主实体类型。
        tail (str): 尾实体名称。
        tail_type (str): 尾实体类型。

    Returns:
        str: 模型预测输出结果或错误信息。

    Raises:
        Exception: 运行配置文件读取、写入或子进程调用失败时触发。
    """
    if task != "standard":
        return "[RE任务目前只支持standard模式]"

    base_path = os.path.expanduser(os.path.join(
        os.getenv("DEEPKE_PATH"), "example/re/standard"
    ))
    try:
        input_text = f"n\n{txt}\n{head}\n{head_type}\n{tail}\n{tail_type}\n"
        command = f"cd {base_path} && {os.getenv('CONDA_DEEPKE_PY')}python predict.py"
        process = subprocess.Popen(
            ["bash", "-c", command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=input_text)
        return stdout if process.returncode == 0 else f"[运行失败]\n{stderr}"
    except Exception as e:
        return f"[运行异常] {str(e)}"


def deepke_ae(txt: str, entity: str, attribute_value: str, task: TaskType = 'standard') -> str:
    """执行 DeepKE 的属性抽取（AE）任务。

    Args:
        txt (str): 输入文本。
        entity (str): 目标实体。
        attribute_value (str): 实体的属性或属性值。
        task (TaskType): 任务类型，目前仅支持 'standard'。

    Returns:
        str: 模型预测输出结果或错误信息。

    Raises:
        Exception: 运行配置文件读取、写入或子进程调用失败时触发。
    """
    base_path = os.path.expanduser(os.path.join(
        os.getenv("DEEPKE_PATH"), "example/ae/standard"
    ))
    try:
        input_text = f"n\n{txt}\n{entity}\n{attribute_value}\n"
        command = f"cd {base_path} && {os.getenv('CONDA_DEEPKE_PY')}python predict.py"
        process = subprocess.Popen(
            ["bash", "-c", command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=input_text)
        return stdout if process.returncode == 0 else f"[运行失败]\n{stderr}"
    except Exception as e:
        return f"[运行异常] {str(e)}"


def deepke_ee(txt: str) -> str:
    """执行 DeepKE 的事件抽取（EE）任务。

    包括触发词预测（trigger）与论元预测（role）两个阶段。
    内部会动态修改 `conf/train.yaml` 的任务名并依次执行。

    Args:
        txt (str): 输入文本内容。

    Returns:
        str: 包含触发词与论元预测结果的字符串。

    Raises:
        Exception: 当配置读取、运行脚本或文件访问失败时触发。

    Notes:
        输出包含：
            - 触发词预测结果与内容；
            - 论元预测结果与内容；
        每部分由换行符分隔，便于终端直接阅读。
    """
    base_path = os.path.expanduser(os.path.join(
        os.getenv("DEEPKE_PATH"), "example/ee/standard"
    ))
    trigger_eval_path = os.path.join(base_path, "exp/DuEE/trigger/bert-base-chinese/eval_pred.json")
    trigger_result_path = os.path.join(base_path, "exp/DuEE/trigger/bert-base-chinese/eval_results.txt")
    role_eval_path = os.path.join(base_path, "exp/DuEE/role/bert-base-chinese/eval_pred.json")
    role_result_path = os.path.join(base_path, "exp/DuEE/role/bert-base-chinese/eval_results.txt")
    train_config_path = os.path.join(base_path, "conf/train.yaml")

    # 将输入文本转换为 raw 与 tsv 格式
    input_to_raw_and_tsv(txt,
        os.path.join(base_path, "data/DuEE/raw/duee_dev.json"),
        os.path.join(base_path, "data/DuEE/role/dev.tsv"),
        os.path.join(base_path, "data/DuEE/trigger/dev.tsv")
    )

    try:
        # 第一阶段：Trigger 预测
        with open(train_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["task_name"] = "trigger"
        with open(train_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)
        command = f"cd {base_path} && {os.getenv('CONDA_DEEPKE_EE_PY')}python run.py"
        subprocess.run(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        # 第二阶段：Role 预测
        with open(train_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["task_name"] = "role"
        with open(train_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)
        command = f"cd {base_path} && {os.getenv('CONDA_DEEPKE_EE_PY')}python run.py"
        subprocess.run(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        # 第三阶段：预测阶段
        command = f"cd {base_path} && {os.getenv('CONDA_DEEPKE_EE_PY')}python predict.py"
        subprocess.run(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        # 汇总结果
        with open(trigger_eval_path, "r", encoding="utf-8") as f:
            trigger_data = f.read()
        with open(trigger_result_path, "r", encoding="utf-8") as f:
            trigger_result = f.read()
        with open(role_eval_path, "r", encoding="utf-8") as f:
            role_data = f.read()
        with open(role_result_path, "r", encoding="utf-8") as f:
            role_result = f.read()

        result = (
            "请核对结果，根据分隔一一确定预测内对应原句位置的内容，"
            "B，I，O分别代表起始、中间和无关字。\n"
            "[触发词预测结果]\n"
            f"{trigger_result.strip()}\n\n"
            "[触发词预测内容]\n"
            f"{trigger_data.strip()}\n\n"
            "[论元预测结果]\n"
            f"{role_result.strip()}\n\n"
            "[论元预测内容]\n"
            f"{role_data.strip()}"
        )
        return result
    except Exception as e:
        return f"[运行失败] {str(e)}"
