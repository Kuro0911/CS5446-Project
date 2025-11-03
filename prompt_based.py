# prompt_based.py
from __future__ import annotations
import re, time, random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# -----------------------------
# Optional deps (robust fallback)
# -----------------------------
try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch, F = None, None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# -----------------------------
# Config
# -----------------------------
@dataclass
class PromptDefenseConfig:
    # Embeddings
    emb_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    emb_thresh: float = 0.15          # clamp tiny cosine sims to 0 (denoise)
    # Risk thresholds in [0,1]
    strict_threshold: float = 0.30
    extreme_threshold: float = 0.50
    # Transform toggles
    paraphrase: bool = False          # generally better OFF (can help attackers)
    max_edits: int = 0
    strip_urls: bool = True
    strip_injections: bool = True

# -----------------------------
# Patterns & text blocks
# -----------------------------
_JAILBREAK_MARKERS = [
    r"ignore\s+previous\s+instructions",
    r"override\s+system\s+prompt",
    r"as\s+an\s+uncensored\s+model",
    r"without\s+ethical\s+constraints",
    r"developer\s+mode",
    r"\{\{.*?\}\}",
    r"#\s*system\s*:",
    r"#\s*assistant\s*:",
    r"simulate\s+(?:being|as)",
    r"role\s*play\s+as",
    r"\b(bypass|evade|circumvent)\b",
    r"\bjailbreak\b",
]

MILD_PREFIX = (
    "Safety note: I will keep this discussion high-level and avoid sensitive or harmful details. "
    "If your intent is benign, please clarify as needed.\n\n"
)
STRICT_PREFIX = (
    "Safety notice: The request appears potentially risky. I will NOT provide harmful, illegal, or dangerous instructions. "
    "I can offer safe, high-level context and alternatives instead.\n\n"
)
EXTREME_PREFIX = (
    "Safety override: This request appears unsafe. I will refuse to provide any operational or step-by-step guidance. "
    "If you have a lawful, safe learning goal, please reframe the question accordingly.\n\n"
)

# -----------------------------
# Light paraphraser (optional)
# -----------------------------
_SYNONYM_MAP = {
    "give":   ["provide", "offer"],
    "tell":   ["share", "explain"],
    "hack":   ["compromise"],
    "bypass": ["avoid", "circumvent"],
}

def _paraphrase_light(s: str, k: int) -> str:
    if k <= 0:
        return s
    words = s.split()
    idxs = list(range(len(words)))
    random.shuffle(idxs)
    edits = 0
    for i in idxs:
        token = words[i]
        base = re.sub(r"\W+", "", token).lower()
        if base in _SYNONYM_MAP:
            repl = random.choice(_SYNONYM_MAP[base])
            # replace base (case-insensitive) but preserve punctuation around it
            words[i] = re.sub(base, repl, token, flags=re.IGNORECASE)
            edits += 1
            if edits >= k:
                break
    return " ".join(words)

# -----------------------------
# Sanitizer
# -----------------------------
def _sanitize(s: str, *, strip_urls: bool, strip_injections: bool) -> str:
    x = s
    if strip_urls:
        x = re.sub(r"https?://\S+", "[link]", x)
    if strip_injections:
        for pat in _JAILBREAK_MARKERS:
            x = re.sub(pat, "[removed]", x, flags=re.IGNORECASE)
    x = re.sub(r"\s+", " ", x).strip()
    return x

# -----------------------------
# Embedder with offline fallback
# -----------------------------
class _Embedder:
    """Use ST embeddings if available; otherwise fall back to a keyword risk."""
    def __init__(self, model_name: str, unsafe_vocab: Optional[List[str]] = None):
        self.ok = False
        self.mdl = None
        self.unsafe_vocab = set((unsafe_vocab or []) + [
            "bypass","evade","circumvent","hack","exploit","malware","backdoor",
            "bomb","weapon","poison","deepfake","impersonate","doxx","illegal",
            "keylogger","phishing","ransomware","ddos","trojan"
        ])
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

    def kw_risk(self, s: str) -> float:
        s_low = s.lower()
        hits = sum(1 for w in self.unsafe_vocab if re.search(rf"\b{re.escape(w)}\b", s_low))
        # Map count → [0,1] with a gentle cap
        return max(0.0, min(1.0, hits / 5.0))

# -----------------------------
# Builder
# -----------------------------
def build_prompt_defense(
    safe_seed_prompts: List[str],
    unsafe_seed_prompts: List[str],
    cfg: Optional[PromptDefenseConfig] = None,
) -> Callable[[str], Tuple[str, Optional[str], Dict]]:
    """
    Returns a callable that takes the raw user prompt and returns:
        (rewritten_user_prompt, None, info)
    We intentionally return None for the system prompt so the caller sends only the user message.
    """
    cfg = cfg or PromptDefenseConfig()
    emb = _Embedder(cfg.emb_model)

    # Compute centroids (if embeddings available)
    safe_centroid = None
    unsafe_centroid = None

    if emb.ok and unsafe_seed_prompts:
        E_unsafe = emb.encode(unsafe_seed_prompts)
        if torch is not None and E_unsafe is not None:
            unsafe_centroid = F.normalize(E_unsafe.mean(dim=0, keepdim=True), dim=-1)

    if emb.ok and safe_seed_prompts:
        E_safe = emb.encode(safe_seed_prompts)
        if torch is not None and E_safe is not None:
            safe_centroid = F.normalize(E_safe.mean(dim=0, keepdim=True), dim=-1)

    def _risk(prompt: str) -> float:
        # Comparative risk = max(0, sim_unsafe - sim_safe)
        if emb.ok and (unsafe_centroid is not None or safe_centroid is not None):
            e = emb.encode([prompt])
            if e is None:
                return 0.0
            sim_u = float((e @ unsafe_centroid.T).squeeze().item()) if unsafe_centroid is not None else 0.0
            sim_s = float((e @ safe_centroid.T).squeeze().item())   if safe_centroid   is not None else 0.0
            # Denoise tiny sims
            if sim_u < cfg.emb_thresh: sim_u = 0.0
            if sim_s < cfg.emb_thresh: sim_s = 0.0
            return max(0.0, min(1.0, sim_u - sim_s))
        # Fallback: keyword risk
        return emb.kw_risk(prompt)

    def defense(user_prompt: str) -> Tuple[str, Optional[str], Dict]:
        """
        Only rewrites the USER prompt (no system prompt).
        The rewrite prepends a safety wrapper + sanitized content.
        """
        t0 = time.time()
        r0 = _risk(user_prompt)

        # Sanitize (strip URLs/jailbreak markers)
        x = _sanitize(
            user_prompt,
            strip_urls=cfg.strip_urls,
            strip_injections=cfg.strip_injections,
        )

        # Optional paraphrase (usually keep off)
        if cfg.paraphrase and cfg.max_edits > 0:
            x = _paraphrase_light(x, cfg.max_edits)

        # Choose prepend wrapper by risk (no system message is used)
        if r0 >= cfg.extreme_threshold:
            wrapped = EXTREME_PREFIX + f"User request:\n{x}"
            action = "PREPEND_EXTREME"
        elif r0 >= cfg.strict_threshold:
            wrapped = STRICT_PREFIX + f"User request:\n{x}"
            action = "PREPEND_STRICT"
        else:
            wrapped = MILD_PREFIX + x
            action = "PREPEND_MILD"

        r1 = _risk(wrapped)
        info = {
            "risk_before": r0,
            "risk_after": r1,
            "action": action,
            "latency_ms": int((time.time() - t0) * 1000),
        }
        return wrapped, None, info  # <-- No system prompt

    return defense