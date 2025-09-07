import os
import subprocess
import yaml
from convert_to_tsv import input_to_raw_and_tsv
from typing import Literal

TaskType = Literal["standard", "few-shot", "multimodal", "documental"]

def deepke_ner(task: TaskType, txt: str) -> str:
    if task != "standard":
        return "[NER任务目前只支持standard模式]"

    base_path = os.path.expanduser(os.path.join(
        os.getenv("DEEPKE_PATH"), "example/ner/standard"
    ))
    config_path = os.path.join(base_path, "conf/predict.yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["text"] = txt
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)

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
    base_path = os.path.expanduser(os.path.join(
        os.getenv("DEEPKE_PATH"), "example/ee/standard"
    ))
    trigger_eval_path = os.path.join(base_path, "exp/DuEE/trigger/bert-base-chinese/eval_pred.json")
    trigger_result_path = os.path.join(base_path, "exp/DuEE/trigger/bert-base-chinese/eval_results.txt")
    role_eval_path = os.path.join(base_path, "exp/DuEE/role/bert-base-chinese/eval_pred.json")
    role_result_path = os.path.join(base_path, "exp/DuEE/role/bert-base-chinese/eval_results.txt")
    train_config_path = os.path.join(base_path, "conf/train.yaml")

    input_to_raw_and_tsv(txt,
        os.path.join(base_path, "data/DuEE/raw/duee_dev.json"),
        os.path.join(base_path, "data/DuEE/role/dev.tsv"),
        os.path.join(base_path, "data/DuEE/trigger/dev.tsv")
    )

    try:
        # trigger
        with open(train_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["task_name"] = "trigger"
        with open(train_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)
        command = f"cd {base_path} && {os.getenv('CONDA_DEEPKE_EE_PY')}python run.py"
        subprocess.run(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        # role
        with open(train_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["task_name"] = "role"
        with open(train_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)
        command = f"cd {base_path} && {os.getenv('CONDA_DEEPKE_EE_PY')}python run.py"
        subprocess.run(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        # predict
        command = f"cd {base_path} && {os.getenv('CONDA_DEEPKE_EE_PY')}python predict.py"
        subprocess.run(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        with open(trigger_eval_path, "r", encoding="utf-8") as f:
            trigger_data = f.read()
        with open(trigger_result_path, "r", encoding="utf-8") as f:
            trigger_result = f.read()
        with open(role_eval_path, "r", encoding="utf-8") as f:
            role_data = f.read()
        with open(role_result_path, "r", encoding="utf-8") as f:
            role_result = f.read()

        result = (
            "请核对结果，根据分隔一一确定预测内对应原句位置的内容，B，I，O分别代表起始，中间和无关字。\n"
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
