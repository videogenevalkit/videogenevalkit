"""ViCLIP-L/14 backbone loader for viclip-score.

ViCLIP (InternVideo) is trained on video-text pairs, so it scores T2V
alignment more faithfully than per-frame CLIP. The model code is vendored
under ``metrics/third_party/viclip`` (MIT, OpenGVLab/InternVideo).

Weight resolution order:
  1. ``$VIDEVALKIT_VICLIP_WEIGHTS``                     [explicit override]
  2. ``$VIDEVALKIT_CACHE_HOME/viclip/<file>``           [our fetch cache]
  3. ``~/.cache/vbench/ViCLIP/<file>``                  [shared vbench cache]

Tokenizer vocab (``bpe_simple_vocab_16e6.txt.gz``) is resolved alongside the
weight in the same directory. ``fetch_viclip_weights`` downloads both from the
OpenGVLab HF mirror when absent.
"""

from __future__ import annotations

import os
from pathlib import Path

WEIGHT_FILE = "ViClip-InternVid-10M-FLT.pth"
TOKENIZER_FILE = "bpe_simple_vocab_16e6.txt.gz"

_HF_WEIGHT_URL = (
    "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/"
    "ViClip-InternVid-10M-FLT.pth"
)
_HF_TOKENIZER_URL = (
    "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/"
    "bpe_simple_vocab_16e6.txt.gz"
)


def _cache_home() -> Path:
    return Path(
        os.environ.get("VIDEVALKIT_CACHE_HOME", Path.home() / ".cache" / "videvalkit")
    )


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("VIDEVALKIT_VICLIP_WEIGHTS")
    if env:
        p = Path(env)
        dirs.append(p.parent if p.suffix else p)
    dirs.append(_cache_home() / "viclip")
    dirs.append(Path.home() / ".cache" / "vbench" / "ViCLIP")
    return dirs


def resolve_viclip_dir() -> Path | None:
    """Return the first dir containing the ViCLIP weight, else None."""
    for d in _candidate_dirs():
        if (d / WEIGHT_FILE).is_file():
            return d
    return None


def fetch_viclip_weights() -> Path:
    """Download weight + tokenizer into the fetch cache; return that dir."""
    import urllib.request

    found = resolve_viclip_dir()
    if found is not None:
        return found
    dst = _cache_home() / "viclip"
    dst.mkdir(parents=True, exist_ok=True)
    for url, name in ((_HF_WEIGHT_URL, WEIGHT_FILE), (_HF_TOKENIZER_URL, TOKENIZER_FILE)):
        target = dst / name
        if not target.is_file():
            urllib.request.urlretrieve(url, target)  # noqa: S310 (trusted HF host)
    return dst


def load_viclip(device: str = "cuda"):
    """Load the vendored ViCLIP-L model + tokenizer.

    Returns (model, tokenizer). Raises FileNotFoundError if weights are absent
    and not downloadable.
    """
    from ..third_party.viclip.simple_tokenizer import SimpleTokenizer
    from ..third_party.viclip.viclip import ViCLIP

    wdir = resolve_viclip_dir()
    if wdir is None:
        raise FileNotFoundError(
            f"ViCLIP weight {WEIGHT_FILE!r} not found in any of "
            f"{[str(d) for d in _candidate_dirs()]}. Set "
            f"$VIDEVALKIT_VICLIP_WEIGHTS to its directory, or call "
            f"videvalkit.metrics.backbones.viclip_l.fetch_viclip_weights() "
            f"to download it from OpenGVLab/VBench_Used_Models."
        )
    tok_path = wdir / TOKENIZER_FILE
    if not tok_path.is_file():
        raise FileNotFoundError(
            f"ViCLIP tokenizer {TOKENIZER_FILE!r} not found next to weight in {wdir}"
        )
    from ..utils.device import resolve_device
    dev = resolve_device(device)
    tokenizer = SimpleTokenizer(str(tok_path))
    model = ViCLIP(tokenizer=tokenizer, pretrain=str(wdir / WEIGHT_FILE))
    model = model.to(dev).eval()
    return model, tokenizer
