# dataset.py
import os
from typing import Callable, Dict, List, Optional, Tuple, Literal, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from datasets import load_dataset
    _HAS_HF = True
except Exception:
    _HAS_HF = False

DEFAULT_XSTEST_CSV = "xstest_prompts.csv"

class SimpleTextDataset(Dataset):

    def __init__(self, df: pd.DataFrame):
        required = {"id", "prompt", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        self.df = df.reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        return {
            "id": int(row["id"]),
            "prompt": str(row["prompt"]),
            "label": str(row["label"]),
        }

def load_xstest_minimal(
    csv_path: str = DEFAULT_XSTEST_CSV,
    *,
    shuffle: bool = False,
    seed: int = 42,
) -> pd.DataFrame:

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"XSTest CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)

    keep = ["id", "prompt", "label"]
    for c in keep:
        if c not in df.columns:
            raise ValueError(f"XSTest CSV must contain column: {c}")

    out = df[keep].copy()
    out["prompt"] = out["prompt"].astype(str).str.strip()
    out = out[out["prompt"].str.len() > 0]

    lab = out["label"].astype(str).str.lower().str.strip()
    lab = lab.map({"safe": "safe", "unsafe": "unsafe"})
    out["label"] = lab.fillna("safe") 

    out = out.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    if shuffle:
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out["id"] = out.index.astype(int)
    return out[["id", "prompt", "label"]]

HF_REPO = "TrustAIRLab/in-the-wild-jailbreak-prompts"
WildSplit = Literal[
    "jailbreak_2023_05_07",
    "jailbreak_2023_12_25",
    "regular_2023_05_07",
    "regular_2023_12_25",
]

def _ensure_hf():
    if not _HAS_HF:
        raise RuntimeError("Hugging Face 'datasets' is not installed. Run: pip install datasets")

def _normalize_hf_df_minimal(raw_df: pd.DataFrame, label_value: str) -> pd.DataFrame:

    df = raw_df.copy()
    text_col = next((c for c in ["prompt", "content", "text", "raw_prompt"] if c in df.columns), None)
    if text_col is None:
        raise ValueError(f"Could not find a prompt/text column in {list(df.columns)}")

    out = pd.DataFrame()
    out["prompt"] = df[text_col].astype(str).str.strip()
    out = out[out["prompt"].str.len() > 0]
    out = out.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    out["label"] = "unsafe" if label_value == "unsafe" else "safe"
    out["id"] = out.index.astype(int)
    return out[["id", "prompt", "label"]]

def load_in_the_wild_minimal(
    split: WildSplit = "jailbreak_2023_12_25",
    *,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:

    _ensure_hf()
    ds = load_dataset(HF_REPO, name=split, split="train")  # IMPORTANT: name=split
    raw_df = ds.to_pandas()
    label_value = "unsafe" if split.startswith("jailbreak_") else "safe"
    out = _normalize_hf_df_minimal(raw_df, label_value)
    if max_rows is not None and len(out) > max_rows:
        out = out.sample(max_rows, random_state=42).reset_index(drop=True)
        out["id"] = out.index.astype(int)
    return out

def load_in_the_wild_pair_minimal(
    jailbreak_split: WildSplit = "jailbreak_2023_12_25",
    regular_split:   WildSplit = "regular_2023_12_25",
    *,
    max_unsafe: Optional[int] = 200,
    max_safe: Optional[int] = 200,
) -> pd.DataFrame:

    df_unsafe = load_in_the_wild_minimal(jailbreak_split, max_rows=max_unsafe)
    df_safe   = load_in_the_wild_minimal(regular_split,   max_rows=max_safe)
    df = pd.concat([df_unsafe, df_safe], axis=0, ignore_index=True)
    df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    df["id"] = df.index.astype(int)
    return df[["id", "prompt", "label"]]

def combine_minimal(
    dfs: List[pd.DataFrame],
    *,
    dedup: bool = True,
    shuffle: bool = True,
    seed: int = 42,
) -> pd.DataFrame:

    if not dfs:
        return pd.DataFrame(columns=["id", "prompt", "label"])
    df = pd.concat(dfs, axis=0, ignore_index=True)
    if dedup:
        df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["id"] = df.index.astype(int)
    return df[["id", "prompt", "label"]]

def load_combined_minimal(
    xstest_csv: str = DEFAULT_XSTEST_CSV,
    *,
    jailbreak_split: WildSplit = "jailbreak_2023_12_25",
    regular_split:   WildSplit = "regular_2023_12_25",
    max_unsafe: Optional[int] = 300,
    max_safe: Optional[int] = 300,
    shuffle: bool = True,
    seed: int = 42,
) -> SimpleTextDataset:

    df_xs = load_xstest_minimal(xstest_csv)
    df_wild = load_in_the_wild_pair_minimal(
        jailbreak_split=jailbreak_split,
        regular_split=regular_split,
        max_unsafe=max_unsafe,
        max_safe=max_safe,
    )
    df_all = combine_minimal([df_xs, df_wild], dedup=True, shuffle=shuffle, seed=seed)
    return SimpleTextDataset(df_all)

def split_train_val_test(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    r_train, r_val, r_test = ratios
    if abs(r_train + r_val + r_test - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * r_train)
    n_val = int(n * r_val)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    
    for d in (train, val, test):
        d["id"] = d.reset_index(drop=True).index.astype(int)
    return train, val, test

def balanced_sample(
    df: pd.DataFrame,
    max_per_class: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:

    if max_per_class is None:
        return df
    out = []
    for lab in ["unsafe", "safe"]:
        sub = df[df["label"] == lab]
        if len(sub) > max_per_class:
            sub = sub.sample(max_per_class, random_state=seed)
        out.append(sub)
    res = pd.concat(out, axis=0, ignore_index=True)
    res = res.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    res["id"] = res.index.astype(int)
    return res[["id", "prompt", "label"]]

def get_seed_sets_for_steering(
    df: pd.DataFrame,
    n_safe: int = 32,
    n_unsafe: int = 32,
    seed: int = 7,
) -> Tuple[List[str], List[str]]:

    rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=seed).index
    safe_prompts = df[df["label"] == "safe"].loc[:, "prompt"]
    unsafe_prompts = df[df["label"] == "unsafe"].loc[:, "prompt"]
    safe_list = safe_prompts.sample(min(n_safe, len(safe_prompts)), random_state=seed).tolist()
    unsafe_list = unsafe_prompts.sample(min(n_unsafe, len(unsafe_prompts)), random_state=seed).tolist()
    return safe_list, unsafe_list

def default_chat_template(prompt: str) -> str:
    system_prefix = "You are a helpful, safe, and honest assistant."
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prefix}\n<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

def make_collate_fn(
    tokenizer,
    max_length: int = 2048,
    make_label_tensor: bool = False,
) -> Callable[[List[Dict]], Dict[str, torch.Tensor]]:

    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        if hasattr(tokenizer, "apply_chat_template"):
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": "You are a helpful, safe, and honest assistant."},
                     {"role": "user", "content": p}],
                    add_generation_prompt=True, tokenize=False
                ) for p in prompts
            ]
        else:
            texts = [default_chat_template(p) for p in prompts]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc["ids"] = torch.tensor([int(item["id"]) for item in batch], dtype=torch.long)
        labels_raw = [item["label"] for item in batch]
        enc["labels_raw"] = labels_raw 

        if make_label_tensor:
            enc["labels_tensor"] = torch.tensor([1 if l == "unsafe" else 0 for l in labels_raw], dtype=torch.long)

        return enc
    return collate

def make_dataloader(
    ds: Dataset,
    tokenizer=None,
    batch_size: int = 4,
    max_length: int = 2048,
    num_workers: int = 0,
    shuffle: bool = False,
    make_label_tensor: bool = False,
) -> DataLoader:

    if tokenizer is None:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    collate_fn = make_collate_fn(tokenizer, max_length=max_length, make_label_tensor=make_label_tensor)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )