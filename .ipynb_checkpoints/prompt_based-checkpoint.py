from __future__ import annotations
import re, time, random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

try:
    import torch
except Exception:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

@dataclass
class PromptDefenseConfig:
    emb_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    emb_thresh: float = 0.42
    strict_threshold: float = 0.30 
    extreme_threshold: float = 0.60
    paraphrase: bool = True
    max_edits: int = 2
    strip_urls: bool = True
    strip_injections: bool = True


_JAILBREAK_MARKERS = [
    r"ignore\s+previous\s+instructions",
    r"override\s+system\s+prompt",
    r"as\s+an\s+uncensored\s+model",
    r"without\s+ethical\s+constraints",
    r"developer\s+mode",
    r"\{\{.*?\}\}",
]

_MILD = (
    "You are a helpful, honest, and safety-conscious assistant. "
    "Answer the user's questions accurately and responsibly while maintaining a respectful and neutral tone. "
    "If a request could be unsafe, gently remind the user of relevant safety or ethical considerations before proceeding."
)

_STRICT = (
    "You are a security-focused assistant whose priority is to prevent harm or misuse. "
    "If a user's request appears risky, unethical, or could cause harm, clearly refuse to comply. "
    "Instead, explain why the action may be unsafe and offer a safe, educational, or lawful alternative that helps the user achieve their goal responsibly."
)

_EXTREME = (
    "SAFETY OVERRIDE ACTIVE: You must refuse any request that could lead to harm, illegal activity, or unethical behavior, even if uncertainty exists. "
    "Provide only high-level, non-sensitive, and non-operational information. "
    "Do not generate instructions, code, or detailed procedures related to restricted, dangerous, or ambiguous topics under any circumstance."
)

def _sanitize(s: str, *, strip_urls: bool, strip_injections: bool) -> str:
    x = s
    if strip_urls:
        x = re.sub(r"https?://\S+", "[link]", x)
    if strip_injections:
        for pat in _JAILBREAK_MARKERS:
            x = re.sub(pat, "[removed]", x, flags=re.IGNORECASE)
    x = re.sub(r"\s+", " ", x).strip()
    return x

_SYNONYM_MAP = {
    "give": ["provide", "offer"],
    "tell": ["share", "explain"],
    "hack": ["breach", "compromise"],
    "bypass": ["evade", "circumvent"],
}

def _paraphrase_light(s: str, k: int) -> str:
    if k <= 0:
        return s
    words = s.split()
    idxs = list(range(len(words)))
    random.shuffle(idxs)
    edits = 0
    for i in idxs:
        base = re.sub(r"\W+", "", words[i]).lower()
        if base in _SYNONYM_MAP:
            repl = random.choice(_SYNONYM_MAP[base])
            words[i] = words[i].replace(base, repl)
            edits += 1
            if edits >= k:
                break
    return " ".join(words)

class _Embedder:
    def __init__(self, model_name: str):
        self.ok = False
        self.mdl = None
        if SentenceTransformer:
            try:
                self.mdl = SentenceTransformer(model_name)
                self.ok = True
            except Exception:
                self.ok = False

    def encode(self, texts: List[str]):
        if not self.ok:
            return None
        v = self.mdl.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        if torch is not None and not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        return v


def build_prompt_defense(
    safe_seed_prompts: List[str],
    unsafe_seed_prompts: List[str],
    cfg: Optional[PromptDefenseConfig] = None,
) -> Callable[[str], Tuple[str, str, Dict]]:

    cfg = cfg or PromptDefenseConfig()
    emb = _Embedder(cfg.emb_model)

    centroid = None
    if emb.ok and unsafe_seed_prompts:
        E = emb.encode(unsafe_seed_prompts)
        if torch is not None and E is not None:
            centroid = torch.nn.functional.normalize(E.mean(dim=0, keepdim=True), dim=-1)

    def _risk(prompt: str) -> float:
        if centroid is None:
            return 0.0
        e = emb.encode([prompt])
        if e is None:
            return 0.0
        sim = float((e @ centroid.T).squeeze().item())  # [-1,1]
        return max(0.0, min(1.0, sim))                  # clamp to [0,1]

    def _choose_system(r: float) -> str:
        if r >= cfg.extreme_threshold:
            return _EXTREME
        if r >= cfg.strict_threshold:
            return _STRICT
        return _MILD

    def defense(user_prompt: str) -> Tuple[str, str, Dict]:
        t0 = time.time()
        r0 = _risk(user_prompt)
        sys = _choose_system(r0)

        x = _sanitize(user_prompt, strip_urls=cfg.strip_urls, strip_injections=cfg.strip_injections)
        if cfg.paraphrase:
            x = _paraphrase_light(x, cfg.max_edits)

        r1 = _risk(x)
        sys = _choose_system(max(r0, r1))
        info = {"risk_before": r0, "risk_after": r1, "latency_ms": int((time.time()-t0)*1000)}
        return x, sys, info

    return defense