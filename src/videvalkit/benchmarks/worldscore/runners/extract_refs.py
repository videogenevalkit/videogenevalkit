"""Extract the WorldScore reference images for our 50 static + 100 dynamic entries.

Match by content_list (static) and prompt (dynamic) to the upstream parquet rows.
Saves to $VIDEVALKIT_WS_REFS_OUT/{prompt_id}.png (default: ~/.cache/videvalkit/smoke-data/worldscore/videos/cogvideox-5b/refs/).
"""
import json, io, glob
import os as _os
from pathlib import Path
import pandas as pd
from PIL import Image

WS_DS = Path("/tmp/worldscore_dataset")
OUT_REF = Path(_os.environ.get("VIDEVALKIT_WS_REFS_OUT", str(Path.home() / ".cache" / "videvalkit" / "smoke-data" / "worldscore" / "videos" / "cogvideox-5b" / "refs")))
OUT_REF.mkdir(parents=True, exist_ok=True)

STATIC_ENT = Path(_os.environ.get("VIDEVALKIT_WS_STATIC_ENT_PROMPTS", str(Path.home() / ".cache" / "videvalkit" / "smoke-data" / "worldscore" / "prompts" / "static-entry.jsonl")))
DYNAMIC    = Path(_os.environ.get("VIDEVALKIT_WS_DYNAMIC_PROMPTS", str(Path.home() / ".cache" / "videvalkit" / "smoke-data" / "worldscore" / "prompts" / "dynamic.jsonl")))


def load_parquets(split):
    files = sorted(glob.glob(str(WS_DS / split / "train-*.parquet")))
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def normalize_list(lst):
    """Convert numpy.ndarray to tuple of strings for hashing."""
    if hasattr(lst, "tolist"): lst = lst.tolist()
    return tuple(str(x) for x in lst)


def main():
    # ---- STATIC ----
    print("loading static parquet...")
    df_s = load_parquets("static")
    print(f"  {len(df_s)} static entries in upstream dataset")

    # Index by content_list tuple
    df_s["content_key"] = df_s["content_list"].apply(normalize_list)
    s_index = {row["content_key"]: row for _, row in df_s.iterrows()}

    n_static_ok, n_static_missing = 0, 0
    for line in STATIC_ENT.open():
        e = json.loads(line)
        key = tuple(str(x) for x in e["content_list"])
        if key in s_index:
            row = s_index[key]
            img_bytes = row["image"]["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            out = OUT_REF / f"{e['entry_id']}.png"
            img.save(out)
            n_static_ok += 1
        else:
            n_static_missing += 1
            print(f"  MISSING static: {e['entry_id']}  content={key[:80]!s}...")
    print(f"static: {n_static_ok} saved, {n_static_missing} missing")

    # ---- DYNAMIC ----
    print("\nloading dynamic parquet...")
    df_d = load_parquets("dynamic")
    print(f"  {len(df_d)} dynamic entries in upstream dataset")

    # Index by prompt string
    df_d["prompt_key"] = df_d["prompt"]
    d_index = {row["prompt_key"]: row for _, row in df_d.iterrows()}

    n_dyn_ok, n_dyn_missing = 0, 0
    for line in DYNAMIC.open():
        e = json.loads(line)
        prompt = e["prompt"]
        if prompt in d_index:
            row = d_index[prompt]
            img_bytes = row["image"]["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            out = OUT_REF / f"{e['prompt_id']}.png"
            img.save(out)
            n_dyn_ok += 1
        else:
            n_dyn_missing += 1
            print(f"  MISSING dynamic: {e['prompt_id']}  prompt={prompt[:80]}...")
    print(f"dynamic: {n_dyn_ok} saved, {n_dyn_missing} missing")

    print(f"\nrefs saved to {OUT_REF}")


if __name__ == "__main__":
    main()
