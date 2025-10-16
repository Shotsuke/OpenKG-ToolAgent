"""DeepKE 事件抽取 (EE) 任务时的输入预处理工具
"""

import json
import hashlib


def generate_id(text: str) -> str:
    """为输入文本生成唯一的 MD5 哈希 ID。

    Args:
        text (str): 输入文本。

    Returns:
        str: 文本对应的 MD5 哈希字符串。
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def text_to_0x02_sequence(text: str, fill: str = 'O') -> tuple[str, str, str, str]:
    """将文本转化为以 0x02 (ASCII 中的文本分隔符) 分隔的序列。

    同时生成对应的标签序列与索引序列，这种格式常用于事件抽取任务中的 token 对齐。

    Args:
        text (str): 输入文本。
        fill (str, optional): 标签填充值，默认使用 'O' 表示“非事件”标记。

    Returns:
        tuple[str, str, str, str]: 
            - 文本序列
            - 标签序列
            - 触发词或论元标签序列
            - 字符索引序列
    """
    return (
        '\x02'.join(text),
        '\x02'.join([fill] * len(text)),
        '\x02'.join([fill] * len(text)),
        '\x02'.join([str(i) for i in range(len(text))])
    )


def write_raw_file(text: str, raw_path: str) -> None:
    """将输入文本写入 raw 格式文件，用于 DeepKE 的事件抽取数据读取。

    文件内容包含两个字段：
    - "text": 原始输入文本；
    - "id": 文本对应的 MD5 哈希值。

    Args:
        text (str): 输入文本。
        raw_path (str): 输出文件路径。
    """
    data = {"text": text, "id": generate_id(text)}
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def write_single_sentence_trigger_tsv(text: str, tsv_path: str) -> None:
    """生成触发词识别 (trigger detection) 的 TSV 文件。

    文件包含三列：
    - text_a: 以 0x02 分隔的文本；
    - label: 对应的 BIO 标签序列（此处全为 'O'）；
    - index: 样本编号（此处固定为 0）。

    Args:
        text (str): 输入文本。
        tsv_path (str): 输出 TSV 文件路径。
    """
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write("text_a\tlabel\tindex\n")
        text_a = '\x02'.join(text)
        label = '\x02'.join(['O'] * len(text))
        f.write(f"{text_a}\t{label}\t0\n")


def write_single_sentence_role_tsv(text: str, tsv_path: str) -> None:
    """生成论元识别 (role labeling) 的 TSV 文件。

    文件包含四列：
    - text_a: 以 0x02 分隔的文本；
    - label: BIO 标签序列（此处全为 'O'）；
    - trigger_tag: 触发词标签序列（此处全为 'O'）；
    - index: 样本编号（此处固定为 0）。

    Args:
        text (str): 输入文本。
        tsv_path (str): 输出 TSV 文件路径。
    """
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write("text_a\tlabel\ttrigger_tag\tindex\n")
        text_a = '\x02'.join(text)
        label = '\x02'.join(['O'] * len(text))
        trigger_tag = '\x02'.join(['O'] * len(text))
        f.write(f"{text_a}\t{label}\t{trigger_tag}\t0\n")


def input_to_raw_and_tsv(text: str, raw_path: str, tsv_role_path: str, tsv_trigger_path: str) -> None:
    """综合调用多个函数，将输入文本同时生成 raw 与两类 TSV 文件。

    通常用于 DeepKE EE 模型推理阶段的数据准备。

    Args:
        text (str): 输入文本。
        raw_path (str): 输出 raw 文件路径。
        tsv_role_path (str): 输出 role TSV 文件路径。
        tsv_trigger_path (str): 输出 trigger TSV 文件路径。
    """
    write_raw_file(text, raw_path)
    write_single_sentence_role_tsv(text, tsv_role_path)
    write_single_sentence_trigger_tsv(text, tsv_trigger_path)
    print(f"已生成：\n- raw: {raw_path}\n")


# 示例
# if __name__ == "__main__":
#     input_text = "振华三部曲的《暗恋橘生淮南》终于定档了。"
#     input_to_raw_and_tsv(input_text, "user_raw.json", "user_role_tsv.tsv", "user_trigger_tsv.tsv")
