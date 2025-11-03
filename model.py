import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception:
    hf_hub_download = None
    list_repo_files = None

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

DEFAULT_MODELS = {
    "aligned": "meta-llama/Llama-3.1-8B-Instruct",
    "unaligned": "dphn/dolphin-2.9.1-llama-3-8b",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
auth_token = "HF_TOKEN"

_PREFERRED_Q4K_ORDER = ("Q4_K_M", "Q4_K_S", "Q4_K_L", "Q4_K")
_ENV_LOCAL_GGUF = "HF_GGUF_LOCAL_PATH"


@dataclass
class GenerateConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    stop: Optional[List[str]] = None


class ModelWrapper:
    def __init__(self, model, tokenizer, *, backend: str = "hf"):
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        if backend == "hf":
            self.device = getattr(model, "device", torch.device(DEVICE))
            self.dtype = next(model.parameters()).dtype
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    @property
    def hf(self):
        return self.model

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: Optional[bool] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        if do_sample is None:
            do_sample = temperature > 0.0

        if self.backend == "hf":
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if stop:
                text = _truncate_on_stop(text, stop)
            return text.strip()

        n_predict = max_new_tokens
        t = max(0.0, float(temperature))
        top_p_llama = float(top_p)

        res = self.model(
            prompt,
            max_tokens=n_predict,
            temperature=t,
            top_p=top_p_llama,
            stop=stop or [],
            use_mlock=False,
            echo=False,
        )
        text = res["choices"][0]["text"]
        if stop:
            text = _truncate_on_stop(text, stop)
        return text.strip()

    def attach_hidden_hooks(
        self,
        layers: Union[str, Iterable[int]],
        fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        if self.backend != "hf":
            raise NotImplementedError(
                "attach_hidden_hooks is only supported for HF transformer models, "
                "not for GGUF/llama.cpp backends."
            )

        blocks = _get_transformer_blocks(self.model)
        if isinstance(layers, str):
            if layers.lower() != "all":
                raise ValueError("layers must be 'all' or an iterable of indices")
            idxs = range(len(blocks))
        else:
            idxs = list(layers)

        def _wrap(_fn):
            def _hook(module, inputs, output):
                return _fn(output)
            return _hook

        self.detach_hooks()
        for i in idxs:
            h = blocks[i].register_forward_hook(_wrap(fn))
            self._hook_handles.append(h)

    def detach_hooks(self):
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()


def load_model(
    name_or_key: str = "aligned",
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    device_map: Union[str, dict, None] = "auto",
    auth_token: Optional[str] = None,
):
    model_id = DEFAULT_MODELS.get(name_or_key, name_or_key)
    device = device or DEVICE
    dtype = dtype or DTYPE
    token = auth_token or os.getenv("HF_TOKEN", None) or globals().get("auth_token", None)

    if model_id.strip().lower().endswith(".gguf"):
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is required to load GGUF models. "
                "Install with: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122"
            )

        local_path = os.getenv(_ENV_LOCAL_GGUF, "").strip()
        if local_path and os.path.isfile(local_path):
            gguf_path = local_path
        elif os.path.isfile(model_id):
            gguf_path = model_id
        else:
            if hf_hub_download is None or list_repo_files is None:
                raise RuntimeError(
                    "huggingface_hub is required to auto-download GGUF files. "
                    "Install with: pip install huggingface_hub"
                )
            files = list_repo_files(repo_id=model_id.split("/")[0] + "/" + model_id.split("/")[1], use_auth_token=token)
            candidates = [f for f in files if f.lower().endswith(".gguf") and "q4_k" in f.lower()]
            selected = None
            for pref in _PREFERRED_Q4K_ORDER:
                for f in candidates:
                    if pref.lower() in f.lower():
                        selected = f
                        break
                if selected:
                    break
            if not selected and candidates:
                selected = candidates[0]
            if not selected:
                raise RuntimeError("No Q4_K*.gguf file found in the repo.")
            gguf_path = hf_hub_download(repo_id=model_id, filename=selected, use_auth_token=token)

        n_ctx = int(os.getenv("LLAMACPP_CTX", "8192"))
        n_threads = int(os.getenv("LLAMACPP_THREADS", str(os.cpu_count() or 8)))
        n_gpu_layers = int(os.getenv("LLAMACPP_N_GPU_LAYERS", "1"))

        llama = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            logits_all=False,
            verbose=False,
        )

        class _LlamaCppTokenizerAdapter:
            eos_token = "</s>"
            pad_token = "</s>"

        tokenizer = _LlamaCppTokenizerAdapter()
        return ModelWrapper(llama, tokenizer, backend="llama")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()

    return ModelWrapper(model, tokenizer, backend="hf")


def _get_transformer_blocks(model) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    for attr in ("transformer", "gpt_neox", "model"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "layers"):
            return list(m.layers)
    raise RuntimeError("Could not locate transformer blocks for hook attachment.")


def _truncate_on_stop(text: str, stop: List[str]) -> str:
    cut = len(text)
    for s in stop:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]