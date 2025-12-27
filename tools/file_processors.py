# tools/file_pipeline.py
import os
import csv
import re
import ast
from pathlib import Path
from dotenv import load_dotenv
from deepke import deepke_ner, deepke_ae

load_dotenv()
OUTPUT_DIR = Path(os.path.expanduser(os.getenv("MCP_OUTPUT_DIR", "./outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_path(path: str) -> Path:
    """
    将路径转换为绝对路径：
    - 展开 ~
    - 如果是相对路径，基于当前工作目录转换为绝对路径
    """
    p = Path(os.path.expanduser(path))
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p

def run_ner(txt: str, task: str = "standard") -> str:
    return deepke_ner(txt=txt, task=task)

def run_ae(txt: str, entity: str, attribute_value: str, task: str = "standard") -> str:
    return deepke_ae(txt=txt, entity=entity, attribute_value=attribute_value, task=task)


def parse_ner_output(raw: str):
    match = re.search(r"\[(.*)\]", raw, re.S)
    if not match:
        return []
    return ast.literal_eval("[" + match.group(1) + "]")


def align_ner_to_text(text: str, ner_pairs):
    tokens = list(text)
    labels = ["O"] * len(tokens)
    idx = 0
    for i, ch in enumerate(tokens):
        if idx < len(ner_pairs) and ch == ner_pairs[idx][0]:
            labels[i] = ner_pairs[idx][1]
            idx += 1
    return list(zip(tokens, labels))


def process_ner_file(input_path: Path, task: str) -> str:
    output_path = OUTPUT_DIR / f"{input_path.stem}.ner.txt"
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            text = line.strip()
            if not text:
                continue
            raw = run_ner(txt=text, task=task)
            pairs = parse_ner_output(raw)
            aligned = align_ner_to_text(text, pairs)
            for token, label in aligned:
                fout.write(f"{token} {label}\n")
            fout.write("\n")
    return str(output_path)


def parse_ae_output(raw: str):
    match = re.search(r'在句中属性为："(.+?)"', raw)
    return match.group(1) if match else "Unknown"


def process_ae_file(input_path: Path, task: str) -> str:
    output_path = OUTPUT_DIR / f"{input_path.stem}.ae.csv"
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames + ["attribute"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            raw = run_ae(
                txt=row.get("txt", ""),
                entity=row.get("entity", ""),
                attribute_value=row.get("attribute_value", ""),
                task=task,
            )
            row["attribute"] = parse_ae_output(raw)
            writer.writerow(row)
    return str(output_path)


def dispatch_tool(tool: str, input_path: str, task: str = "standard") -> str:
    """
    根据工具类型调用对应函数
    """
    from pathlib import Path
    input_path = Path(input_path)
    input_path = normalize_path(input_path)
    if tool == "ae":
        return process_ae_file(input_path, task)
    elif tool == "ner":
        return process_ner_file(input_path, task)
    else:
        return f"[错误] 不支持的工具: {tool}"
