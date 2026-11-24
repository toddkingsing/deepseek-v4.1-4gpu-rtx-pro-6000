#!/usr/bin/env python3
"""Offline verifier for the four-GPU DeepSeek-V4.1-Flash-1017 repository.

Checks numeric invariants, prefill reductions, capacity/quality gates,
sanitization hygiene, and strict 4-GPU-only content isolation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "data/2026-11-23-four-gpu-speed10.json").read_text(encoding="utf-8"))
CFG = json.loads((ROOT / "configs/2026-11-23-four-gpu-speed10.json").read_text(encoding="utf-8"))

# 1. Core 4-GPU configuration invariants
assert CFG["tp"] == 4 and CFG["pp"] == 1
assert CFG["kv_GiB_per_gpu"] == 11.0 and CFG["max_num_seqs"] == 16 and CFG["max_model_len"] == 1048576
assert CFG["arguments"]["--kv-cache-memory-bytes"] == "11811160064"
assert CFG["environment"]["VLLM_MOET_PREFILL_PIPE"] == "0"
assert CFG["environment"]["VLLM_MOET_PREFILL_SP"] == "1"
assert CFG["environment"]["VLLM_MOET_PREFILL_SP_RS"] == "nccl"
assert CFG["environment"]["VLLM_MOET_PREFILL_SP_GATHER"] == "mxfp8"
assert CFG["environment"]["VLLM_LANGCHAO_BMM_MAX_T"] == "48"
assert CFG["environment"]["VLLM_MOET_DRAFTER_LM_HEAD_MXFP8"] == "0"
assert CFG["image_content_id"] == DATA["runtime_identity"]["image_content_id"]
assert CFG["inference_fingerprint"] == DATA["runtime_identity"]["inference_fingerprint"]

# 2. Stage 2 single-process GIL prefill breakthrough checks
c1 = DATA["stage2_ab_screening"]["candidates"]["cand1_base_pipe2nccl"]
c2 = DATA["stage2_ab_screening"]["candidates"]["cand2_pipe0_sp_mxfp8"]
red_163k = 1.0 - c2["prefill_163k_s"] / c1["prefill_163k_s"]
red_1m = 1.0 - c2["prefill_1m_s"] / c1["prefill_1m_s"]
assert 0.39 <= red_163k <= 0.41, f"163K reduction out of range: {red_163k}"
assert 0.27 <= red_1m <= 0.29, f"1M reduction out of range: {red_1m}"
assert c2["c1_geo_tps"] > c1["c1_geo_tps"]
assert c2["peak_vram_mib"] < c1["peak_vram_mib"] - 3000

# 3. Stage 3 production speed, capacity & anti-OOM checks
s3 = DATA["stage3_qualified_production_kv11g"]
assert s3["short_prompt_t1_c1"]["c1_median_tps"]["code"] > 370.0
assert s3["short_prompt_t0"]["c1_median_tps"]["chinese"] > 310.0

cap = s3["concurrency_and_capacity"]
assert cap["single_1x1m"]["passed"] and cap["single_1x1m"]["steady_decode_tps"] > 450.0
for key, expected_c in (
    ("concurrent_4x1m", 4),
    ("concurrent_5x1m", 5),
    ("concurrent_6x1m", 6),
    ("concurrent_10x500k", 10),
):
    item = cap[key]
    assert item["passed"] and item["sessions"] == expected_c and item["retrievals_passed"] == expected_c
    assert item["cold_prefix_hits"] == 0 and item["preemptions"] == 0

ov = cap["overflow_8x1m_zero_oom"]
assert ov["passed"] and ov["completed_ok"] == 8 and ov["retrievals_passed"] == 8
assert ov["allocator_or_oom_warnings"] == 0 and ov["preemptions"] == 0

mx = cap["mixed_c4x128k_plus_1m"]
assert mx["passed"] and mx["recovery_window_agg_tps"] > mx["baseline_window_agg_tps"]

# 4. Quality & hardware telemetry gates
q = s3["qualification"]
assert q["gsm8k_200_1024_budget"]["passed"] == 197 and q["gsm8k_200_1024_budget"]["truncated"] == 0
assert q["quality_chat"]["passed"] == 29 and q["quality_low"]["passed"] == 30
assert q["long_quality"]["passed"] == 12 and q["application_checks"]["passed"] == 10
assert q["short_vision"]["passed"] == 2 and q["long_vision"]["passed"] == 4
assert q["public_https_acceptance"]["passed"] == 10 and q["public_cold_cancel"]["passed"] == 3
assert all(q["post_promotion_three_protocol_check"].values())

hw = s3["hardware_telemetry"]
assert hw["peak_vram_mib"] + hw["free_vram_headroom_mib"] == hw["total_vram_mib_per_gpu"]
assert hw["swap_current_bytes"] == 0 and hw["swap_max_bytes"] == 0 and hw["allocator_retries"] == 0

# 5. Asset existence check
for asset in (
    "assets/deepseek-v41-4gpu-hero-20261124.jpg",
    "assets/deepseek-v41-4gpu-architecture-20261124.png",
    "assets/deepseek-v41-4gpu-evolution-20261124.png",
    "assets/deepseek-v41-4gpu-decode-prefill-20261124.png",
    "assets/deepseek-v41-4gpu-concurrency-capacity-20261124.png",
):
    assert (ROOT / asset).is_file(), f"Missing asset: {asset}"

# 6. Sanitization & strict 4-GPU-only isolation scan
forbidden_secrets = [
    re.compile(r"\b10\.168\.\d+\.\d+\b"),
    re.compile(r"\b192\.168\.\d+\.\d+\b"),
    re.compile(r"\b64\.64\.236\.12\b"),
    re.compile(r"/opt/", re.I),
    re.compile(r"D:\\server", re.I),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"api\.kingsing\.com", re.I),
]
forbidden_8gpu_mixing = [
    re.compile(r"\bTP4[x×]PP2\b", re.I),
    re.compile(r"\beight[- ]gpu\b", re.I),
    re.compile(r"\b8\s*[x×]\s*RTX\b", re.I),
    re.compile(r"八卡"),
    re.compile(r"\b24[x×]1M\b", re.I),
    re.compile(r"\b36\s*GiB\b", re.I),
    re.compile(r"Speed9", re.I),
]

check_files = [
    "README.md",
    "docs/four-gpu-deployment-en.md",
    "docs/four-gpu-deployment-zh.md",
    "data/2026-11-23-four-gpu-speed10.json",
    "configs/2026-11-23-four-gpu-speed10.json",
]
for rel in check_files:
    txt = (ROOT / rel).read_text(encoding="utf-8")
    for pat in forbidden_secrets:
        m = pat.search(txt)
        assert m is None, f"Sanitization violation in {rel}: {m.group(0)}"
    for pat in forbidden_8gpu_mixing:
        m = pat.search(txt)
        assert m is None, f"8-GPU mixing violation in 4-GPU repo ({rel}): {m.group(0)}"

print("PASS: 4-GPU Speed10 invariants, prefill gains, 6x1M/10x500K/8x1M capacity, quality, telemetry, sanitization & strict 4-GPU isolation verified.")
