"""Qwen2.5-Instruct 4bit 로드 + chat 헬퍼 (한국어 지원 생성 LLM)."""

from __future__ import annotations

from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_llm(model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
    llm.eval()
    return tok, llm


@torch.no_grad()
def chat(
    tok,
    llm,
    system: str,
    user: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(llm.device)
    out = llm.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=0.9,
        pad_token_id=tok.eos_token_id,
    )
    gen_tokens = out[0][enc["input_ids"].shape[1] :]
    return tok.decode(gen_tokens, skip_special_tokens=True).strip()
