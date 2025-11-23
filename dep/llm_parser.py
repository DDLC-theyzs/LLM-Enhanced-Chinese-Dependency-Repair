# llm.py
"""
封装硅基流动 Qwen/Qwen3-VL-32B-Instruct 的调用,用于依存句法 CoNLL-U 纠错。

依赖:
    pip install openai

环境变量:
    SILICONFLOW_API_KEY = "你的硅基流动 API Key"
"""

import os
import json
from typing import Optional, Dict, Any

from openai import OpenAI, APIError

#你的siliconflow API key
SILICON_BASE_URL = "https://api.siliconflow.cn/v1"
#调用的LLM，可以在 https://cloud.siliconflow.cn/me/models 选择自己需要的模型
DEFAULT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


class QwenLLM:
    """封装 LLM 调用的简单类。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = SILICON_BASE_URL,
    ) -> None:

        api_key = "sk-iesdcpjpzqjyxwznnyqfuonqomowrckljbhqqcapimrkmtvi"

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def correct_conllu(self, sentence: str, conllu_text: str) -> Dict[str, Any]:
        """
        使用 Qwen 模型对 CoNLL-U 依存结果进行纠错。

        参数
        ----
        sentence : str
            原始中文句子(未分词的原文)。
        conllu_text : str
            对应句子的 CoNLL-U 文本(每个 token 一行,10 列,制表符分隔)。

        返回
        ----
        Dict[str, Any]
            结构化结果,包含:
            {
                "modified": bool,
                "has_uncertainty": bool,
                "modification_reason": str,
                "uncertainty_reason": str,
                "suggestion": str,
                "final_conllu": str,   # 最终要写回文件的 CoNLL-U
                "raw_response": str    # LLM 原始输出,方便调试
            }
        """
        system_prompt = (
            "你是一个依存句法结构的解析与审查助手。\n"
            "你的任务是: 根据给定的原句与初始 CoNLL-U 依存结构,判断该结构是否需要修改,并提供修改后的(或原始) CoNLL-U。\n"
            "\n"
            "【输出格式要求】\n"
            "你必须严格输出一个 JSON 对象,包含以下字段:\n"
            "- \"modified\": 布尔值,表示你是否建议修改依存结构\n"
            "- \"has_uncertainty\": 布尔值,表示你是否对结构判断存在不确定性\n"
            "- \"modification_reason\": 若未建议修改,必须填 \"N/A\"\n"
            "- \"uncertainty_reason\": 若无不确定性,必须填 \"N/A\"\n"
            "- \"suggestion\": 给人工审查的友好说明文本\n"
            "- \"final_conllu\": 最终要写入文件的 CoNLL-U 内容(字符串)\n"
            "  - 若 has_uncertainty = true,则必须返回原始 CoNLL-U\n"
            "  - 若 modified = false,则必须返回原始 CoNLL-U\n"
            "  - 若 modified = true 且 has_uncertainty = false,则返回你修改过的 CoNLL-U\n"
            "\n"
            "【行为规则】\n"
            "1. 不得输出句子内容。句子由系统管理,不能重复或改写。\n"
            "2. 不得修改 ID、FORM,不得增删 token,只能在 HEAD(第7列)和 DEPREL(第8列)上修改。\n"
            "3. 不得改变 token 的数量或顺序。\n"
            "4. 若 has_uncertainty = true,则你必须:\n"
            "   - 设置 modified = false\n"
            "   - 返回原始 CoNLL-U 作为 final_conllu\n"
            "5. 确保最终 CoNLL-U 仍然是合法的依存树:\n"
            "   - 只有一个 root(HEAD=0 且 DEPREL=root 的 token 恰好一个)\n"
            "   - HEAD 必须是 0..N 范围内的整数,不要引入环\n"
            "6. 在 JSON 外禁止输出任何内容(不能输出解释、说明、句子原文)。\n"
            "你的任务是一个“结构判断 + 结构微调器”,而不是生成任意文本。\n"
        )

        user_prompt = (
            "请根据以下信息判断依存结构是否需要修改,并按系统要求返回 JSON。\n\n"
            f"原句: {sentence}\n\n"
            "初始 CoNLL-U 结构:\n"
            f"{conllu_text}\n\n"
            "请严格遵守系统提示:\n"
            "- 判断是否需要修改依存结构\n"
            "- 判断是否存在不确定性\n"
            "- 如果不修改或不确定,返回原始 CoNLL-U\n"
            "- 如果修改,返回修改后的 CoNLL-U\n"
            "- 仅输出 JSON,不要输出任何 JSON 外的字符或文字\n"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,  # 稳定一点,减少乱改
            )
        except APIError as e:
            # 这里可以根据需要做日志记录或重试,这里先直接抛出
            raise RuntimeError(f"调用 Qwen API 失败: {e}") from e

        content = resp.choices[0].message.content

        if not isinstance(content, str):
            raise RuntimeError("LLM 返回内容不是字符串,需检查 API 调用或模型配置。")

        raw_response = content

        # 清理可能的 ``` 包裹
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            # 过滤掉以 ``` 开头的行
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            content = "\n".join(lines).strip()

        # 尝试解析 JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 如果解析失败,做一个保守 fallback: 认为未修改且不确定,直接返回原始 CoNLL-U
            return {
                "modified": False,
                "has_uncertainty": True,
                "modification_reason": "N/A",
                "uncertainty_reason": "JSON 解析失败,已保留原始 CoNLL-U。",
                "suggestion": "模型输出非 JSON 格式,建议人工检查该句。",
                "final_conllu": conllu_text,
                "raw_response": raw_response,
            }

        # 做一些字段兜底处理
        modified = bool(data.get("modified", False))
        has_uncertainty = bool(data.get("has_uncertainty", False))
        modification_reason = data.get("modification_reason", "N/A") or "N/A"
        uncertainty_reason = data.get("uncertainty_reason", "N/A") or "N/A"
        suggestion = data.get("suggestion", "") or "N/A"
        final_conllu = data.get("final_conllu", "").strip()

        # 若 has_uncertainty = True,则强制视为未修改,并回退到原始 CoNLL-U
        if has_uncertainty:
            modified = False
            final_conllu = conllu_text

        # 若 final_conllu 为空,也回退
        if not final_conllu:
            final_conllu = conllu_text

        return {
            "modified": modified,
            "has_uncertainty": has_uncertainty,
            "modification_reason": modification_reason,
            "uncertainty_reason": uncertainty_reason,
            "suggestion": suggestion,
            "final_conllu": final_conllu,
            "raw_response": raw_response,
        }
