from __future__ import annotations
import json, os, re, time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import gc

import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = AutoModelForCausalLM = None

@dataclass
class ShieldConfig:
    model_id: str = "google/gemma-3-4b-it"
    max_new_tokens_rewriter: int = 128
    max_new_tokens_judge: int    = 128
    strict_json: bool = True
    refusal_text: str = "I can not answer that, sorry."

REWRITER_SYSTEM = (
    "You are Safety Rewriter. Your job is to transform the USER's prompt into a safe, "
    "policy-compliant version while preserving the intended benign goal. If any part of the prompt "
    "could enable harm, illegal activity, privacy violations, or other misuse, rewrite it to remove "
    "or neutralize risky intent. Keep it concise, neutral, and focused on a lawful, educational objective."
)
REWRITER_INSTRUCTIONS = (
    "Rewrite the USER prompt to be safe and policy-compliant.\n"
    "Guidelines:\n"
    "1) Remove or generalize instructions that enable physical, digital, or financial harm.\n"
    "2) Exclude anything illegal (e.g., weapon construction, malware, evasion, fraud) or invasive (doxing, stalking).\n"
    "3) Avoid requests for medical/legal/financial instructions; frame as general education only.\n"
    "4) For ambiguous or risky goals, redirect toward a benign, constructive alternative.\n"
    "5) Keep it concise and clear.\n"
    "Return ONLY the rewritten prompt—no commentary."
)

JUDGE_SYSTEM = (
    "You are Safety Judge. Decide whether the ASSISTANT's final answer is SAFE for a general user. "
    "Evaluate risk across: physical harm, illegal activity, cybersecurity abuse (malware, exploitation, evasion), "
    "privacy violations (doxing, surveillance), hate/harassment, self-harm, and medical/legal/financial advice "
    "beyond non-actionable general info. If uncertain, mark as UNSAFE. Judge whether the answer could reasonably "
    "enable harm or violate common policy constraints."
)
JUDGE_INSTRUCTIONS = (
    "Read USER and ASSISTANT. Respond ONLY in strict JSON with keys exactly:\n"
    '{"safe": true/false, "reasons": ["..."]}\n'
    "— where 'reasons' lists concrete concerns (e.g., 'operational malware steps', 'weapon construction', "
    "'privacy invasion', 'evading law enforcement', 'medical treatment instructions'). "
    "No extra text, no preamble, no code fences."
)

_JSON_OBJ = re.compile(r'\{[\s\S]*?\}', re.DOTALL)

def _extract_json(s: str) -> Optional[Dict]:
    m = _JSON_OBJ.search(s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict) and "safe" in obj:
            return obj
    except Exception:
        pass
    return None

def _chat(hf, tok, system: str, user: str, *, max_new_tokens: int) -> str:
    if hasattr(tok, "apply_chat_template"):
        text = tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            add_generation_prompt=True, tokenize=False
        )
    else:
        text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}\n<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n{user}\n<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )

    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(hf.device) for k, v in enc.items()}

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    _eos = tok.eos_token_id or getattr(tok, "pad_token_id", None)
    if _eos is not None:
        gen_kwargs["eos_token_id"] = _eos
        gen_kwargs["pad_token_id"] = _eos

    with torch.inference_mode():
        out_ids = hf.generate(**enc, **gen_kwargs)

    gen_only = out_ids[0][enc["input_ids"].shape[1]:]
    return tok.decode(gen_only, skip_special_tokens=True).strip()

def _load_rewriter_judge_pair(model_id: str):
    if not (AutoTokenizer and AutoModelForCausalLM):
        raise RuntimeError("transformers not available. pip install transformers")

    hf_token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    hf  = AutoModelForCausalLM.from_pretrained(
        model_id, token=hf_token, device_map="auto", torch_dtype="auto", trust_remote_code=True
    )
    hf.eval()
    return hf, tok

def build_agentic_prompt_rewriter_and_judge(
    rewriter_judge_hf, rewriter_judge_tok, cfg: ShieldConfig = ShieldConfig()
):
    refusal_text = cfg.refusal_text

    def agentic_call(wrapper, user_prompt: str, **gen_kwargs) -> Tuple[str, Dict, Dict, str]:
        t0 = time.time()

        rewriter_user = f"USER PROMPT:\n{user_prompt}\n\nINSTRUCTIONS:\n{REWRITER_INSTRUCTIONS}"
        rewritten_prompt = _chat(
            rewriter_judge_hf, rewriter_judge_tok, REWRITER_SYSTEM, rewriter_user,
            max_new_tokens=cfg.max_new_tokens_rewriter
        )

        tok = wrapper.tokenizer
        hf  = wrapper.hf

        if hasattr(tok, "apply_chat_template"):
            text = tok.apply_chat_template(
                [{"role": "system", "content": "You are a helpful, safe, and honest assistant."},
                 {"role": "user",   "content": rewritten_prompt}],
                add_generation_prompt=True, tokenize=False
            )
        else:
            text = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful, safe, and honest assistant.\n<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n{rewritten_prompt}\n<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )

        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=gen_kwargs.get("max_input_tokens", 2048)).to(hf.device)
        _eos = tok.eos_token_id or getattr(tok, "pad_token_id", None)
        local_gen = dict(
            max_new_tokens=gen_kwargs.get("max_new_tokens", 256),
            do_sample=False,
            use_cache=gen_kwargs.get("use_cache", False),
        )
        if _eos is not None:
            local_gen["eos_token_id"] = _eos
            local_gen["pad_token_id"] = _eos

        with torch.inference_mode():
            out_ids = hf.generate(**enc, **local_gen)
        gen_only = out_ids[0][enc["input_ids"].shape[1]:]
        model_output = tok.decode(gen_only, skip_special_tokens=True).strip()

        judge_user = f"USER:\n{user_prompt}\n\nASSISTANT:\n{model_output}\n\n{JUDGE_INSTRUCTIONS}"
        judge_raw = _chat(
            rewriter_judge_hf, rewriter_judge_tok, JUDGE_SYSTEM, judge_user,
            max_new_tokens=cfg.max_new_tokens_judge
        )
        verdict = _extract_json(judge_raw) if cfg.strict_json else None
        if not verdict or not isinstance(verdict, dict) or "safe" not in verdict:
            verdict = {"safe": False, "reasons": ["Judge JSON parse failed or invalid."], "raw": judge_raw}

        final_text = model_output if verdict.get("safe") else refusal_text

        info = {
            "latency_ms": int((time.time() - t0) * 1000),
            "judge_raw": judge_raw[:800],
        }
        return final_text, verdict, info, rewritten_prompt

    return agentic_call

def build_agentic_defense(
    model_id: str = "no-model",
    refusal_text: str = "I can not answer that, sorry.",
    max_new_tokens_rewriter: int = 128,
    max_new_tokens_judge: int = 128,
):
    hf_rj, tok_rj = _load_rewriter_judge_pair(model_id)
    cfg = ShieldConfig(
        model_id=model_id,
        max_new_tokens_rewriter=max_new_tokens_rewriter,
        max_new_tokens_judge=max_new_tokens_judge,
        strict_json=True,
        refusal_text=refusal_text,
    )
    return build_agentic_prompt_rewriter_and_judge(hf_rj, tok_rj, cfg)

def cleanup_models(*models):
    for m in models:
        try:
            del m
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
