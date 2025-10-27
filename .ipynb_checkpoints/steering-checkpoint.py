# steering_strict.py
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

def _apply_chat_template(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": "You are a helpful, safe, and honest assistant."},
            {"role": "user",  "content": prompt},
        ]
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful, safe, and honest assistant.\n<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

@torch.inference_mode()
def _last_token_layer_hiddens(model, tokenizer, prompt: str, max_length: int = 2048) -> List[torch.Tensor]:
    text = _apply_chat_template(tokenizer, prompt)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    device = next(model.parameters()).device
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    use_autocast = torch.cuda.is_available() and device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=use_autocast):
        out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # [emb, layer1, layer2, ...]
    return [hs[i][0, -1, :].detach() for i in range(1, len(hs))]

def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = v.norm()
    return v if n == 0 else v / (n + eps)

def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

@torch.inference_mode()
def _hidden_to_logits(model, h_last: torch.Tensor) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    h = h_last.to(device=device, dtype=dtype, non_blocking=True)
    if hasattr(model, "model") and hasattr(model.model, "norm") and model.model.norm is not None:
        h = model.model.norm(h)
    logits = model.lm_head(h)
    return logits

@torch.inference_mode()
def _layer_logits_for_prompt(model, tokenizer, prompt: str, max_length: int) -> List[torch.Tensor]:
    hs = _last_token_layer_hiddens(model, tokenizer, prompt, max_length)
    return [_hidden_to_logits(model, h) for h in hs]

def _softmax_mass_over_mask(logits: torch.Tensor, mask: torch.Tensor) -> float:
    p = logits.float().softmax(dim=-1)
    return float((p * mask).sum().item())

def _tokenize_variants(tokenizer, words: Iterable[str]) -> List[int]:
    ids: List[int] = []
    seen = set()
    for w in words:
        for variant in {w, w.lower(), w.upper(), w.capitalize()}:
            toks = tokenizer(variant, add_special_tokens=False)["input_ids"]
            for t in toks:
                if t not in seen:
                    seen.add(t); ids.append(t)
            toks2 = tokenizer(" " + variant, add_special_tokens=False)["input_ids"]
            for t in toks2:
                if t not in seen:
                    seen.add(t); ids.append(t)
    return ids

def _build_vocab_mask(vocab_size: int, token_ids: List[int], device: torch.device) -> torch.Tensor:
    m = torch.zeros(vocab_size, device=device, dtype=torch.float32)
    if token_ids:
        idx = torch.tensor([i for i in token_ids if 0 <= i < vocab_size], device=device, dtype=torch.long)
        if idx.numel() > 0:
            m.index_fill_(0, idx, 1.0)
    return m

def _canonicalize_prompt(s: str) -> str:
    s = s.replace("\u200b", "").replace("\u200c", "")
    s = re.sub(r"[\u00AD\u200E\u200F]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _simple_jailbreak_heuristics(s: str) -> float:
    s = _canonicalize_prompt(s)
    patterns = [
        r"ignore (all )?(previous|prior) (instructions|rules)",
        r"do(n'?| )?t follow (the )?(rules|policy|guidelines)",
        r"\bDAN\b|\bdo anything now\b",
        r"jail\s*break|j@ilbreak|jail\-?br[e3]ak",
        r"bypass (the )?(safety|filter|guard|policy)",
        r"as a (fiction|roleplay|hypothetical)",
        r"without (any )?(restrictions?|limitations?)",
        r"output (the )?steps (for|to) (illegal|dangerous|harmful)",
    ]
    score = 0.0
    for p in patterns:
        if re.search(p, s):
            score += 1.0
    return max(0.0, min(1.0, score / 3.0))

@dataclass
class ASVProfile:
    picked_layers: List[int]
    vectors_by_layer: Dict[int, torch.Tensor]
    weights_by_layer: Dict[int, float]
    unsafe_ref_logits_by_layer: Dict[int, torch.Tensor]
    safe_ref_logits_by_layer: Dict[int, torch.Tensor]
    top_tokens_by_layer: Dict[int, List[int]]

class AdaptiveSafetyVectorSteerer:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_length: int = 2048,
        layer_top_pct: float = 0.30,
        top_k_tokens: int = 32,
        step: float = 0.12,
        beta: float = 6.0,
        alpha_center: float = 0.5,
        preserve_norm: bool = True,
        pairwise_sample: Optional[int] = None,
    ):
        self.m = model
        self.tok = tokenizer
        self.max_length = max_length
        self.layer_top_pct = float(max(0.05, min(0.95, layer_top_pct)))
        self.top_k_tokens = int(max(1, top_k_tokens))
        self.step = float(step)
        self.beta = float(beta)
        self.alpha_center = float(alpha_center)
        self.preserve_norm = bool(preserve_norm)
        self.pairwise_sample = pairwise_sample

        p = next(self.m.parameters())
        self.device = p.device
        self.dtype = p.dtype

        self.profile: Optional[ASVProfile] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        # dynamic runtime
        self._last_risk: float = 0.0
        self._risk_mode_active: bool = False
        self._gen_token_counter: int = 0

        self._K_positions: int = 12

        # hysteresis (strict)
        self._tau_high: float = 0.35
        self._tau_low: float = 0.20
        self._alpha_max: float = 1.0
        self._alpha_min: float = 0.0
        self._cooldown_tokens: int = 24
        self._cooldown_counter: int = 0

        self._unsafe_token_ids: List[int] = []
        self._refusal_token_ids: List[int] = []
        self._unsafe_mask: Optional[torch.Tensor] = None
        self._refusal_mask: Optional[torch.Tensor] = None

        self._lambda_unsafe_max: float = 8.0
        self._mu_refusal_max: float  = 6.0
        self._logit_topN: int = 128

        self._lm_head_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        self._safety_range: Optional[Tuple[int, int]] = None
        self._danger_range: Optional[Tuple[int, int]] = None
        self._target_range: Optional[List[int]] = None

    @torch.inference_mode()
    def fit(self, safe_prompts: Sequence[str], unsafe_prompts: Sequence[str]) -> ASVProfile:
        assert len(safe_prompts) > 0 and len(unsafe_prompts) > 0, "Need non-empty safe/unsafe sets."
        self._maybe_build_vocab_masks()

        tmp_logits = _layer_logits_for_prompt(self.m, self.tok, safe_prompts[0], self.max_length)
        n_layers = len(tmp_logits); vocab_dim = tmp_logits[0].shape[-1]
        del tmp_logits
        device, dtype = self.device, self.dtype

        def _mean_layer_logits(prompts: Sequence[str]) -> List[torch.Tensor]:
            acc = [torch.zeros(vocab_dim, device=device, dtype=dtype) for _ in range(n_layers)]
            for ptxt in prompts:
                ll = _layer_logits_for_prompt(self.m, self.tok, ptxt, self.max_length)
                for l in range(n_layers): acc[l].add_(ll[l].to(device))
            for l in range(n_layers): acc[l].div_(len(prompts))
            return acc

        mean_safe_logits   = _mean_layer_logits(safe_prompts)
        mean_unsafe_logits = _mean_layer_logits(unsafe_prompts)

        safety_dissim = []
        for l in range(n_layers):
            c = _cos(_normalize(mean_safe_logits[l]), _normalize(mean_unsafe_logits[l]))
            safety_dissim.append(1.0 - c)

        danger_prior = self._unsafe_mask.clone().to(device=device, dtype=torch.float32) if self._unsafe_mask is not None else torch.zeros(vocab_dim, device=device, dtype=torch.float32)
        if danger_prior.sum() > 0: danger_prior = danger_prior / (danger_prior.norm() + 1e-8)

        danger_sim = []
        for l in range(n_layers):
            p = mean_unsafe_logits[l].float().softmax(dim=-1)
            if danger_prior.sum() == 0: danger_sim.append(0.0)
            else: danger_sim.append(_cos(_normalize(p), danger_prior))

        pct = max(0.05, min(0.95, self.layer_top_pct))
        k = max(1, int(round(pct * n_layers)))
        safety_top = sorted(range(n_layers), key=lambda i: safety_dissim[i], reverse=True)[:k]
        danger_top = sorted(range(n_layers), key=lambda i: danger_sim[i], reverse=True)[:k]
        picked_layers = sorted(set(safety_top).intersection(danger_top))
        if not picked_layers:
            picked_layers = sorted(safety_top[:max(1, n_layers // 20)])

        weights_by_layer: Dict[int, float] = {}
        tot = 0.0
        for l in picked_layers:
            w = max(0.0, safety_dissim[l]) * max(0.0, danger_sim[l])
            weights_by_layer[l] = w
            tot += w
        if tot > 0:
            for l in picked_layers:
                weights_by_layer[l] /= tot
        else:
            for l in picked_layers:
                weights_by_layer[l] = 1.0 / len(picked_layers)

        def _mean_layer_hiddens(prompts):
            acc = [torch.zeros(mean_safe_logits[0].shape[-1]*0 + self.m.lm_head.weight.shape[1], device=device, dtype=dtype) for _ in range(n_layers)]
            first_h = _last_token_layer_hiddens(self.m, self.tok, prompts[0], self.max_length)
            acc = [torch.zeros_like(h.to(device=device, dtype=dtype)) for h in first_h]
            for ptxt in prompts:
                hs = _last_token_layer_hiddens(self.m, self.tok, ptxt, self.max_length)
                for l in range(n_layers): acc[l].add_(hs[l].to(device, dtype=dtype))
            for l in range(n_layers): acc[l].div_(len(prompts))
            return acc

        mean_safe_h = _mean_layer_hiddens(safe_prompts)
        mean_unsafe_h = _mean_layer_hiddens(unsafe_prompts)

        vectors_by_layer: Dict[int, torch.Tensor] = {}
        for l in picked_layers:
            v = _normalize(mean_safe_h[l] - mean_unsafe_h[l])
            vectors_by_layer[l] = v

        unsafe_ref_logits_by_layer = {l: mean_unsafe_logits[l].to("cpu", dtype=torch.float32) for l in picked_layers}
        safe_ref_logits_by_layer   = {l: mean_safe_logits[l].to("cpu", dtype=torch.float32) for l in picked_layers}

        agg_unsafe = torch.zeros(vocab_dim, device=device, dtype=torch.float32)
        for ptxt in unsafe_prompts:
            ll = _layer_logits_for_prompt(self.m, self.tok, ptxt, self.max_length)
            for l in picked_layers:
                agg_unsafe.add_(ll[l].to(device, dtype=torch.float32))
        topk = min(max(5, self.top_k_tokens), vocab_dim)
        top_tokens = torch.topk(agg_unsafe, k=topk, largest=True).indices.tolist()
        top_tokens_by_layer = {l: top_tokens for l in picked_layers}

        self._target_range = picked_layers
        self.profile = ASVProfile(
            picked_layers=picked_layers,
            vectors_by_layer=vectors_by_layer,
            weights_by_layer=weights_by_layer,
            unsafe_ref_logits_by_layer=unsafe_ref_logits_by_layer,
            safe_ref_logits_by_layer=safe_ref_logits_by_layer,
            top_tokens_by_layer=top_tokens_by_layer,
        )
        return self.profile

    @torch.inference_mode()
    def _update_risk_from_logits(self, logits_2d: torch.Tensor):
        if logits_2d.dim() == 1:
            lv = logits_2d
        else:
            lv = logits_2d[0]

        unsafe_mass = 0.0
        if self._unsafe_mask is not None:
            unsafe_mass = _softmax_mass_over_mask(lv, self._unsafe_mask)

        comp_unsafe_sim = 0.0
        if self.profile and self.profile.picked_layers:
            l0 = min(self.profile.picked_layers)
            uref = _normalize(self.profile.unsafe_ref_logits_by_layer[l0].to(self.device, dtype=self.dtype))
            cur = _normalize(lv.to(self.device, dtype=self.dtype))
            comp_unsafe_sim = max(0.0, 1.0 - _cos(cur, uref))

        w1, w2 = 0.7, 0.3
        risk = max(0.0, min(1.0, w1 * comp_unsafe_sim + w2 * unsafe_mass))

        if self._risk_mode_active:
            if (risk <= self._tau_low) and (self._cooldown_counter <= 0):
                self._risk_mode_active = False
            else:
                self._cooldown_counter = max(self._cooldown_counter - 1, 0)
        else:
            if risk >= self._tau_high:
                self._risk_mode_active = True
                self._cooldown_counter = self._cooldown_tokens

        self._last_risk = float(risk)

    @torch.inference_mode()
    def _alpha_from_current_risk(self) -> float:
        r = float(self._last_risk)
        a = 1.0 / (1.0 + math.exp(-self.beta * (r - self.alpha_center)))
        return float(max(self._alpha_min, min(self._alpha_max, a)))

    @torch.inference_mode()
    def enable(self, prompt_for_alpha: Optional[str] = None, alpha_override: Optional[float] = None):
        assert self.profile is not None, "Call fit(...) first."
        self.disable()

        if alpha_override is not None:
            self._last_risk = float(alpha_override)
        elif prompt_for_alpha:
            heur = _simple_jailbreak_heuristics(prompt_for_alpha)
            self._last_risk = max(self._last_risk, float(heur))

        for l in self.profile.picked_layers:
            if self.profile.weights_by_layer.get(l, 0.0) <= 0.0:
                continue
            block = self.m.model.layers[l]
            handle = block.register_forward_hook(self._make_hook_for_layer(l))
            self._handles.append(handle)

        if hasattr(self.m, "lm_head") and self.m.lm_head is not None:
            self._lm_head_handle = self.m.lm_head.register_forward_hook(self._make_lm_head_hook())

        self._gen_token_counter = 0

    @torch.inference_mode()
    def disable(self):
        for h in self._handles:
            try: h.remove()
            except Exception: pass
        self._handles = []
        if self._lm_head_handle is not None:
            try: self._lm_head_handle.remove()
            except Exception: pass
            self._lm_head_handle = None

    def _make_hook_for_layer(self, l: int):
        device, dtype = self.device, self.dtype
        v_l = self.profile.vectors_by_layer[l].to(device=device, dtype=dtype, non_blocking=True)
        w_l = float(self.profile.weights_by_layer.get(l, 0.0))
        base_step = float(self.step)
        preserve = self.preserve_norm
        K = int(max(1, self._K_positions))

        def _get_hidden(out):
            return out[0] if isinstance(out, tuple) else out

        @torch.inference_mode()
        def hook(module, inputs, output):
            if w_l <= 0.0:
                return output
            h = _get_hidden(output)
            if not isinstance(h, torch.Tensor) or h.dim() != 3:
                return output
            bs, T, H = h.shape
            if T == 0: return output

            alpha_p = self._alpha_from_current_risk()
            risk_flag = 1.0 if self._risk_mode_active else 0.0
            gain = (alpha_p * w_l * base_step) * (0.50 + 0.50 * risk_flag)

            k = min(K, T)
            decays = torch.linspace(1.0, 0.55, steps=k, device=h.device, dtype=h.dtype)
            for i in range(1, k + 1):
                idx = -i
                last = h[:, idx, :]
                if preserve:
                    base_norm = last.norm(dim=-1, keepdim=True)
                delta = (gain * float(decays[i - 1])) * v_l
                last_new = last + delta
                if preserve:
                    last_new = last_new * (base_norm / (last_new.norm(dim=-1, keepdim=True) + 1e-8))
                last.copy_(last_new)
            return output

        return hook

    def _make_lm_head_hook(self):
        unsafe_mask = self._unsafe_mask
        refusal_mask = self._refusal_mask
        topN = int(self._logit_topN)

        @torch.inference_mode()
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return output

            logits = output
            if logits.dim() == 3:
                last = logits[:, -1, :]
            elif logits.dim() == 2:
                last = logits
            else:
                return output

            self._update_risk_from_logits(last)

            self._apply_logit_filters_inplace(last, unsafe_mask, refusal_mask, topN)

            if self._risk_mode_active:
                self._gen_token_counter += 1
                if self._gen_token_counter >= self._cooldown_tokens:
                    self._cooldown_counter = max(self._cooldown_counter - 1, 0)
            return output

        return hook

    @torch.inference_mode()
    def _apply_logit_filters_inplace(self, logits_2d: torch.Tensor,
                                     unsafe_mask: Optional[torch.Tensor],
                                     refusal_mask: Optional[torch.Tensor],
                                     topN: int):
        if logits_2d.numel() == 0:
            return
        r = float(self._last_risk)
        ramp = max(0.0, min(1.0, (r - self._tau_low) / max(1e-6, (self._tau_high - self._tau_low))))
        if self._risk_mode_active:
            ramp = max(ramp, 0.6)

        lam = self._lambda_unsafe_max * ramp
        mu  = self._mu_refusal_max  * ramp
        if lam <= 0 and mu <= 0:
            return

        if logits_2d.dim() == 1:
            logits_2d = logits_2d.unsqueeze(0)

        for i in range(logits_2d.shape[0]):
            row = logits_2d[i]
            p = row.softmax(-1)
            top_vals, top_idx = torch.topk(row, k=min(topN, row.shape[-1]))
            mask_top = torch.zeros_like(row)
            mask_top.index_fill_(0, top_idx, 1.0)

            if unsafe_mask is not None and lam > 0.0:
                penal = lam * p * unsafe_mask * mask_top
                row.sub_(penal)

            if refusal_mask is not None and mu > 0.0:
                boost = mu * (1.0 - p) * refusal_mask * mask_top
                row.add_(boost)

    def steering_context(self, prompt_for_alpha: Optional[str] = None, alpha_override: Optional[float] = None):
        class _Ctx:
            def __init__(self, outer): self.outer = outer
            def __enter__(self):
                self.outer.enable(prompt_for_alpha, alpha_override)
                return self.outer
            def __exit__(self, exc_type, exc, tb):
                self.outer.disable()
        return _Ctx(self)

    def _maybe_build_vocab_masks(self):
        if self._unsafe_mask is not None and self._refusal_mask is not None:
            return
        unsafe_seed = [
            "bomb","explosive","grenade","molotov","napalm","assassinate","kill","poison","harm","gun","rifle",
            "bypass","exploit","xss","sql injection","xss payload","buffer overflow","ransomware","malware","keylogger","backdoor",
            "credit card","ssn","social security","password","private key",
            "make drugs","counterfeit","child","terror","terrorist","fraud",
        ]
        refusal_seed = [
            "cannot","can't","won't","sorry","I cannot","I can't","I will not",
            "I will not help","policy","safety","cannot assist","not able to help",
            "I can provide general information","Here are safer alternatives",
        ]
        try:
            self._unsafe_token_ids = _tokenize_variants(self.tok, unsafe_seed)
            self._refusal_token_ids = _tokenize_variants(self.tok, refusal_seed)
        except Exception:
            self._unsafe_token_ids = []
            self._refusal_token_ids = []

        vocab_size = getattr(self.m.lm_head, "out_features", None)
        if vocab_size is None and hasattr(self.m.lm_head, "weight"):
            vocab_size = self.m.lm_head.weight.shape[0]
        if vocab_size is None:
            vocab_size = int(getattr(self.tok, "vocab_size", 0) or 0)

        self._unsafe_mask  = _build_vocab_mask(vocab_size, self._unsafe_token_ids, self.device)
        self._refusal_mask = _build_vocab_mask(vocab_size, self._refusal_token_ids, self.device)
