# agent.py

"""
agent.py: 串联分句 → 解析 → LLM 校正 → 写入 CoNLL-U 和日志。

使用方式:
    python agent.py
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any

from pre_process import preprocess_file
from nlp_parser import HanLPMultiTaskParser
from llm_parser import QwenLLM


def wrap_conllu_block(sent_id: str, 
                      sentence: str, 
                      conllu_body: str
    ) -> str:

    """
    把单句的 token 行包装成标准的 CoNLL-U 句子块:
    # sent_id = 00001
    # text = 原句...
    1   ...
    2   ...
    
    (句间用空行分隔)
    """
    conllu_body = conllu_body.strip("\n")
    header = f"# sent_id = {sent_id}\n# text = {sentence}\n"
    return header + conllu_body + "\n\n"


def write_one_sentence_conllu(
    file_path: str | Path,
    sent_id: str,
    sentence: str,
    conllu_body: str,
) -> None:
    """
    将一句的 CoNLL-U 写入指定文件(以追加方式).
    """
    block = wrap_conllu_block(sent_id, sentence, conllu_body)
    file_path = Path(file_path)
    with file_path.open("a", encoding="utf-8") as f:
        f.write(block)

def log_result(
    log_path: Path,
    sent_id: str,
    sentence: str,
    raw_conllu: str,
    llm_result: Dict[str, Any],
) -> None:
    """
    以 JSON Lines 形式记录一条 LLM 决策日志。
    """
    record = {
        "sent_id": sent_id,
        "sentence": sentence,
        "modified": llm_result.get("modified", False),
        "has_uncertainty": llm_result.get("has_uncertainty", False),
        "modification_reason": llm_result.get("modification_reason", "N/A"),
        "uncertainty_reason": llm_result.get("uncertainty_reason", "N/A"),
        "suggestion": llm_result.get("suggestion", "N/A"),
        # 可选：把原始/最终 CoNLL-U 也记一份,方便以后排查
        "raw_conllu": raw_conllu,
        "final_conllu": llm_result.get("final_conllu", raw_conllu),
        # 原始模型输出, 排查 prompt 问题时很有用
        "raw_response": llm_result.get("raw_response", ""),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")



# 路径
input_path = "test.txt"
output_path = "output.conllu"
log_path = Path("llm.log")

# 读取文件，预处理并分句
raw_storage: Dict[str, str] = preprocess_file(input_path)

#初始化解析器
nlp_parser = HanLPMultiTaskParser()
print("NLP 解析器初始化完成。")
llm_parser = QwenLLM()
print("LLM 解析器初始化完成。\n")

print(f"共分句：{len(raw_storage)} 句\n")
for k, v in list(raw_storage.items())[0:2]:
    print(f"正在处理第{k}句: {v}\n")

    # 记录当前处理的句子ID和内容
    cur_index = k
    cur_sentence = v

    #调用 NLP 解析器
    result_nlp = nlp_parser.parse(v)
    print(f"nlp结果:\n{result_nlp}\n")

    #调用 LLM 进行校正
    result_llm = llm_parser.correct_conllu(
        sentence=cur_sentence,
        conllu_text=result_nlp
    )
    
    print(f"llm结果:\n{json.dumps(result_llm, ensure_ascii=False, indent=2)}\n")

    #记录日志
    log_result(
        log_path=log_path,
        sent_id=cur_index,
        sentence=cur_sentence,
        raw_conllu=result_nlp,
        llm_result=result_llm
    )

    #写入 CoNLL-U 文件
    write_one_sentence_conllu(
        file_path=output_path,
        sent_id=cur_index,
        sentence=cur_sentence,
        conllu_body=result_llm["final_conllu"]
    )

    print(f"已写入句子 {cur_index} 的 CoNLL-U 到 {output_path}\n")


    

