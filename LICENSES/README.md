# Per-upstream licenses

The toolkit itself is **Apache-2.0** (see `/LICENSE` at the repo root). Each upstream benchmark or model we adapt retains its own license; this directory surfaces those for transparency.

| Upstream | What we use | License | Source |
|---|---|---|---|
| **VBench v1** | Adapter code, prompts, scoring backbones (DINO, CLIP, AMT, ViClip, UMT, GRiT, MUSIQ, tag2text, RAFT, LAION-aesthetic) | Apache-2.0 | https://github.com/Vchitect/VBench |
| **VBench-2.0** | Adapter code, prompts, 18-dim per-prompt registry | Apache-2.0 | https://github.com/Vchitect/VBench (VBench-2.0 subfolder) |
| **Video-Bench** | Adapter prompts (5 alignment + 4 quality), 4-turn rubric | research-use; see upstream | https://github.com/Video-Bench/Video-Bench |
| **WorldJen** | Adapter prompts, PHAS calibrated weights, Phase A/B VQA scaffolding | research-use; see upstream | https://github.com/moonmath-ai/WorldJen-benchmarking-subsystem |
| **WorldScore** | Adapter code, 10 metric backbones (DROID-SLAM, SEA-RAFT, VFIMamba, SAM2, GroundingDINO, CLIP-IQA+, LAION-aesthetic) | research-use; per-backbone licenses | https://github.com/yhw-yhw/WorldScore |
| **T2V-CompBench V2** | Adapter code, prompts, LLaVA-1.6-34B paper-mode subprocess shim, CV stack (GD, SAM-H, Depth-Anything V1, DOT) | research-use; LLaVA non-commercial | https://github.com/KaiyueSun98/T2V-CompBench/tree/V2 |

## Backbone model licenses

| Backbone | License | Notes |
|---|---|---|
| `liuhaotian/llava-v1.6-34b` | LLaMA-derivative; non-commercial research use | Used by T2V-CompBench paper-mode only |
| `lmms-lab/LLaVA-Video-7B-Qwen2` | Qwen2-derivative; research use | Used by VBench-2.0 (12 dims) |
| `Qwen/Qwen2.5-7B-Instruct` | Qwen-research license | Used by VBench-2.0 Complex_Plot judge |
| `Qwen/Qwen2.5-VL-3B-Instruct` | Qwen-research license | Used by VBench-2.0 Instance_Preservation |
| `openai/clip-vit-base-patch16` | MIT | Used by WorldScore content_alignment |
| `LiheYoung/depth_anything_vitl14` | Apache-2.0 | Used by T2V-CompBench spatial |
| GroundingDINO-SwinT-OGC, SAM-H, RAFT, SEA-RAFT, DROID-SLAM, VFIMamba, SAM2 | various open licenses | See respective upstream repos |

When you report numbers from a specific adapter, please cite the upstream paper alongside this toolkit. BibTeX entries in `docs/citations.bib`.
