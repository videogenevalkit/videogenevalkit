# Vendored: ViCLIP (InternVideo)

Source: https://github.com/OpenGVLab/InternVideo (ViCLIP)
License: MIT

Files in this directory (`simple_tokenizer.py`, `viclip.py`, `viclip_text.py`,
`viclip_vision.py`) are vendored byte-for-byte from the upstream ViCLIP module
as redistributed in VBench's `third_party/ViCLIP`. They provide the ViCLIP-L/14
video-text encoder used by the `viclip-score` metric.

Weights (`ViClip-InternVid-10M-FLT.pth`) are NOT vendored; they are fetched at
runtime from `OpenGVLab/VBench_Used_Models` on the Hugging Face Hub. See
`metrics/backbones/viclip_l.py`.
