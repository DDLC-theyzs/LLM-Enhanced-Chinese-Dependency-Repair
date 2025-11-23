# pre_process.py
import re
import json
from pathlib import Path
from typing import Dict, Optional, List


def read_txt(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return text.lstrip("\ufeff")  # 去掉 BOM


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    text = " ".join(lines)

    # 删除所有引号/括号形式的装饰性符号
    text = re.sub(r"[\"\'“”‘’「」『』《》〈〉（）()【】\[\]｛｝]", "", text)

    return text


def split_sentences(text: str) -> List[str]:
    # 把各种空白归一成一个空格，避免乱七八糟的换行、制表符
    text = re.sub(r"\s+", " ", text)

    # 在 “句号/问号/感叹号 + 可选右引号” 后面插入一个特殊分隔标记
    # [。！？!?]          → 中文/英文句末标点
    # [\"'」』】》]?      → 可能跟在后面的右引号、右括号（可选）
    text = re.sub(r"([。！？!?][\"'」』】》]?)", r"\1<SENT_SPLIT>", text)

    # 按标记切分
    parts = text.split("<SENT_SPLIT>")

    # 去掉两端空白 + 过滤空串
    sents = [p.strip() for p in parts if p.strip()]

    return sents


def preprocess_file(path: str, save_json: Optional[str] = None) -> Dict[str, str]:
    """
    返回结构：
    {
        "000001": "句子1",
        "000002": "句子2",
        ...
    }
    """
    text = read_txt(path)
    text = clean_text(text)
    sents = split_sentences(text)

    # 生成 {id: sentence}
    sent_dict = {
        f"{i:05d}": s for i, s in enumerate(sents, start=1)
    }

    if save_json:
        json.dump(sent_dict, open(save_json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    return sent_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    sent_dict = preprocess_file(args.input, save_json=args.output)
    print(f"共分句：{len(sent_dict)}")
    for k, v in list(sent_dict.items())[0:2]:
        print(k, v)
