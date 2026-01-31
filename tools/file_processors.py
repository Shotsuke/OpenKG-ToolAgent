# tools/file_pipeline.py
import os
import csv
import re
import ast
from pathlib import Path
from dotenv import load_dotenv
from deepke import deepke_ner, deepke_ae, deepke_re

load_dotenv()
OUTPUT_DIR = Path(os.path.expanduser(os.getenv("MCP_OUTPUT_DIR", "./outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 类型映射表地址，如果不用cnSchema需要替换为自己的映射表 
TYPE_MAPPING_FILE = Path(
    os.path.expanduser(
        os.path.join(
            os.getenv("DEEPKE_PATH", "~/DeepKE"),
            "example/triple/cnschema/data/type.txt"
        )
    )
)

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

def run_re(txt: str, head: str, head_type: str, tail: str, tail_type: str, task: str = "standard") -> str:
    return deepke_re(task=task, txt=txt, head=head, head_type=head_type, tail=tail, tail_type=tail_type)

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


def parse_re_output(raw: str):
    """
    从 deepke_re 输出中提取关系和置信度，
    如果无法解析，返回 "Unknown"。
    """
    try:
        m = re.search(r'关系为[:：]\s*["“”]?(.+?)["“”]?[，,。]?(\s*置信度.*)?$', raw)
        if m:
            rel = m.group(1).strip()
            conf_m = re.search(r'置信度为[:：]?\s*([0-9]*\.?[0-9]+)', raw)
            confidence = float(conf_m.group(1)) if conf_m else 0.0
            return rel, confidence
        else:
            return "Unknown", 0.0
    except Exception:
        return "Unknown", 0.0
    
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

def process_re_file(input_path: Path, task: str) -> str:
    output_path = OUTPUT_DIR / f"{input_path.stem}.re.csv"
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = (reader.fieldnames or []) + ["relation", "confidence"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            raw = run_re(
                txt=row.get("txt", ""),
                head=row.get("head", ""),
                head_type=row.get("head_type", ""),
                tail=row.get("tail", ""),
                tail_type=row.get("tail_type", ""),
                task=task,
            )
            rel, conf = parse_re_output(raw)
            row["relation"] = rel
            row["confidence"] = conf
            writer.writerow(row)
    return str(output_path)

def process_ee_file(input_path: Path, task: str) -> str:
    
    return "Not implemented yet."


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
    elif tool == "re":
        return process_re_file(input_path, task)
    elif tool == "ee":
        return process_ee_file(input_path, task)
    else:
        return f"[错误] 不支持的工具: {tool}"


def parse_ner_bio(input_path: str, type_mapping_path: str = TYPE_MAPPING_FILE):
    """
    解析 NER BIO 格式文件到对应的实体类型，返回结构化结果：
    [
      {
        "txt": sentence_text,
        "entities": [
            {"text": ent_text, "type": ent_type},
            ...
        ]
      },
      ...
    ]
    """
    input_path = normalize_path(Path(input_path))

    # 读取类型映射表
    label2word = {}
    with open(type_mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                readable, code = line.strip().split()
                label2word[code] = readable

    # 读取 BIO 文件，按句子切分
    sentences = []
    with open(input_path, "r", encoding="utf-8") as f:
        words, labels = [], []
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append((words, labels))
                    words, labels = [], []
                continue
            token, label = line.split()
            words.append(token)
            labels.append(label)
        if words:
            sentences.append((words, labels))

    results = []

    for words, labels in sentences:
        sentence_text = "".join(words)

        entities = []
        cur_ent = ""
        cur_type = ""

        for w, lab in zip(words, labels):
            if lab == "O":
                if cur_ent:
                    entities.append({
                        "text": cur_ent,
                        "type": label2word.get(cur_type, cur_type),
                    })
                    cur_ent, cur_type = "", ""
                continue

            prefix, code = lab.split("-", 1)

            if prefix == "B":
                if cur_ent:
                    entities.append({
                        "text": cur_ent,
                        "type": label2word.get(cur_type, cur_type),
                    })
                cur_ent = w
                cur_type = code

            elif prefix == "I":
                if cur_type == code:
                    cur_ent += w
                else:
                    cur_ent = w
                    cur_type = code

        if cur_ent:
            entities.append({
                "text": cur_ent,
                "type": label2word.get(cur_type, cur_type),
            })

        results.append({
            "txt": sentence_text,
            "entities": entities,
        })

    return results

def ner_bio_to_re_input(input_path: str) -> str:
    """
    将NER BIO格式结果映射到对应的实体类型，并转换为RE输入格式
    input_path: NER结果路径，每行 "字 标签"，句子之间用空行分隔
    """
    ner_path = normalize_path(Path(input_path))
    output_path = OUTPUT_DIR / f"{ner_path.stem}.re_input.csv"

    # 使用 parse_ner_bio 解析 BIO 文件
    parsed = parse_ner_bio(input_path)

    re_rows = []

    for item in parsed:
        sentence_text = item["txt"]
        entities = [
            (ent["text"], ent["type"])
            for ent in item["entities"]
        ]

        # 两两配对生成 RE 输入，目前只将先出现的实体视为head实体然后与后出现的配对，但是可能会漏
        # TODO：是不是可以将每对实体反转后也加进去，然后两种顺序置信度哪个高就选哪个；或者根据cnSchema已有的关系表进行筛选
        for i in range(len(entities)):
            head, head_type = entities[i]
            for j in range(i + 1, len(entities)):
                tail, tail_type = entities[j]
                re_rows.append([
                    sentence_text,
                    head,
                    head_type,
                    tail,
                    tail_type,
                ])

    # 写入 CSV
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["txt", "head", "head_type", "tail", "tail_type"]
        )
        writer.writerows(re_rows)

    return str(output_path)

def ner_bio_to_ae_input(input_path: str) -> str:
    """
    将 NER BIO 格式结果转换为 AE 输入格式：
    (entity, attribute_value, context)
    """
    ner_path = normalize_path(Path(input_path))
    output_path = OUTPUT_DIR / f"{ner_path.stem}.ae_input.csv"

    # 使用 parse_ner_bio 解析 BIO 文件
    parsed = parse_ner_bio(input_path)

    ae_rows = []

    for item in parsed:
        sentence_text = item["txt"]
        entities = [
            (ent["text"], ent["type"])
            for ent in item["entities"]
        ]

        # 两两配对生成 AE 输入，目前只将先出现的实体视为实体然后与后出现的配对，但是可能会漏
        # TODO：是不是可以将每对实体反转后也加进去，然后两种顺序置信度哪个高就选哪个
        # entity 作为主体，attribute_value 作为候选属性值
        for i in range(len(entities)):
            entity, entity_type = entities[i]
            for j in range(len(entities)):
                if i == j:
                    continue

                attr_val, attr_val_type = entities[j]

                ae_rows.append([
                    sentence_text,
                    entity,
                    entity_type,
                    attr_val,
                    attr_val_type,
                ])

    # 写入 CSV
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "txt",
            "entity",
            "entity_type",
            "attribute_value",
            "attribute_value_type",
        ])
        writer.writerows(ae_rows)

    return str(output_path)