from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from .date_align import AlignmentReport, assert_domain_aligned

ROOT = Path(__file__).resolve().parents[2]

# Agriculture (Time-MMD Algriculture spelling)
AGRI_ROOT = ROOT / "data/Algriculture"
AGRI_DATA_PATH = "US_RetailBroilerComposite_Month.csv"
AGRI_CSV = AGRI_ROOT / AGRI_DATA_PATH
DEFAULT_AGRI_CFA_BUNDLE = ROOT / "artifacts/apo_agriculture/runpy_cfa_bundle.pt"

DEFAULT_AGRI_UNIMODAL_CHECKPOINT_SETTING: Optional[str] = None

# Traffic (Time-MMD US VMT monthly)
TRAFFIC_ROOT = ROOT / "data/Traffic"
TRAFFIC_DATA_PATH = "US_VMT_Month.csv"
TRAFFIC_CSV = TRAFFIC_ROOT / TRAFFIC_DATA_PATH
DEFAULT_TRAFFIC_CFA_BUNDLE = ROOT / "artifacts/apo_traffic/runpy_cfa_bundle.pt"

DEFAULT_TRAFFIC_UNIMODAL_CHECKPOINT_SETTING: Optional[str] = None


def _base_monthly_args(
    *,
    model_id_prefix: str,
    root_path: str,
    data_path: str,
    text_injection_mode: str,
) -> SimpleNamespace:
    return argparse.Namespace(
        task_name="long_term_forecast",
        model_id=f"{model_id_prefix}_2021_8_6_PatchTST_BERT_{text_injection_mode}_lr2.0",
        model="PatchTST",
        data="custom",
        root_path=root_path,
        data_path=data_path,
        features="S",
        target="OT",
        freq="h",
        checkpoints=str(ROOT / "checkpoints"),
        seq_len=8,
        label_len=4,
        pred_len=6,
        seasonal_patterns="Monthly",
        inverse=False,
        enc_in=1,
        dec_in=1,
        c_out=1,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        expand=2,
        d_conv=4,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=48,
        num_workers=0,
        batch_size=32,
        llm_model="BERT",
        llm_dim=768,
        llm_layers=6,
        text_path="None",
        type_tag="#F#",
        text_len=4,
        text_max_length=1536,
        prompt_weight=0.1,
        pool_type="avg",
        use_fullmodel=0,
        use_closedllm=0,
        use_text_integrated=1,
        text_injection_mode=text_injection_mode,
        cfa_reduction=8,
        lr_multiplier=2.0,
        des="Exp",
        seed=2021,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices="0",
        use_amp=False,
        text_emb=6,
        init_method="normal",
        learning_rate=0.0002,
        learning_rate2=0.02,
        learning_rate3=0.002,
        learning_rate_weight=0.0002,
        huggingface_token="NA",
    )


def _base_agriculture_args(text_injection_mode: str) -> SimpleNamespace:
    return _base_monthly_args(
        model_id_prefix="Algriculture",
        root_path=str(AGRI_ROOT),
        data_path=AGRI_DATA_PATH,
        text_injection_mode=text_injection_mode,
    )


def resolve_agri_cfa_checkpoint_setting(
    checkpoint_setting: Optional[str] = None,
) -> str:
    """CFA run.py setting folder (for cfa_bundle.pt / checkpoint.pth)."""
    if checkpoint_setting and "_cfa_" in checkpoint_setting:
        return checkpoint_setting
    args = _base_agriculture_args("cfa")
    setting = build_setting_name(args, 0)
    ckpt = ROOT / "checkpoints" / setting / "checkpoint.pth"
    if ckpt.exists():
        return setting
    if checkpoint_setting:
        # User passed unimodal setting; derive CFA sibling by name swap.
        return checkpoint_setting.replace("_unimodal_", "_cfa_")
    raise FileNotFoundError(
        "Agriculture CFA checkpoint not found. Train with "
        "scripts/repro_cfa_agriculture.sh cfa first. Expected: "
        f"{ckpt}"
    )


def resolve_agri_unimodal_checkpoint_setting(
    checkpoint_setting: Optional[str] = None,
) -> str:
    if checkpoint_setting:
        return checkpoint_setting
    if DEFAULT_AGRI_UNIMODAL_CHECKPOINT_SETTING:
        return DEFAULT_AGRI_UNIMODAL_CHECKPOINT_SETTING
    # Discover by matching setting name from args
    args = _base_agriculture_args("unimodal")
    setting = build_setting_name(args, 0)
    ckpt = ROOT / "checkpoints" / setting / "checkpoint.pth"
    if ckpt.exists():
        return setting
    raise FileNotFoundError(
        "Agriculture unimodal checkpoint not found. Train with "
        "scripts/train_agriculture_unimodal.py first. Expected: "
        f"{ckpt}"
    )


def build_agriculture_unimodal_args(
    checkpoint_setting: Optional[str] = None,
) -> SimpleNamespace:
    args = _base_agriculture_args("unimodal")
    args.checkpoint_setting = resolve_agri_unimodal_checkpoint_setting(
        checkpoint_setting
    )
    return args


def build_agriculture_cfa_args(
    checkpoint_setting: Optional[str] = None,
    cfa_bundle_path: Optional[str] = None,
) -> SimpleNamespace:
    args = _base_agriculture_args("cfa")
    args.checkpoint_setting = resolve_agri_unimodal_checkpoint_setting(
        checkpoint_setting
    )
    args.cfa_checkpoint_setting = resolve_agri_cfa_checkpoint_setting(
        checkpoint_setting
    )
    args.cfa_bundle_path = str(cfa_bundle_path or DEFAULT_AGRI_CFA_BUNDLE)
    args.train_epochs = 10
    args.patience = 5
    return args


def _base_traffic_args(text_injection_mode: str) -> SimpleNamespace:
    return _base_monthly_args(
        model_id_prefix="Traffic",
        root_path=str(TRAFFIC_ROOT),
        data_path=TRAFFIC_DATA_PATH,
        text_injection_mode=text_injection_mode,
    )


def resolve_traffic_cfa_checkpoint_setting(
    checkpoint_setting: Optional[str] = None,
) -> str:
    if checkpoint_setting and "_cfa_" in checkpoint_setting:
        return checkpoint_setting
    args = _base_traffic_args("cfa")
    setting = build_setting_name(args, 0)
    ckpt = ROOT / "checkpoints" / setting / "checkpoint.pth"
    if ckpt.exists():
        return setting
    if checkpoint_setting:
        return checkpoint_setting.replace("_unimodal_", "_cfa_")
    raise FileNotFoundError(
        "Traffic CFA checkpoint not found. Train with "
        "scripts/repro_cfa_traffic.sh cfa first. Expected: "
        f"{ckpt}"
    )


def resolve_traffic_unimodal_checkpoint_setting(
    checkpoint_setting: Optional[str] = None,
) -> str:
    if checkpoint_setting:
        return checkpoint_setting
    if DEFAULT_TRAFFIC_UNIMODAL_CHECKPOINT_SETTING:
        return DEFAULT_TRAFFIC_UNIMODAL_CHECKPOINT_SETTING
    args = _base_traffic_args("unimodal")
    setting = build_setting_name(args, 0)
    ckpt = ROOT / "checkpoints" / setting / "checkpoint.pth"
    if ckpt.exists():
        return setting
    raise FileNotFoundError(
        "Traffic unimodal checkpoint not found. Train with "
        "scripts/repro_cfa_traffic.sh unimodal first. Expected: "
        f"{ckpt}"
    )


def build_traffic_unimodal_args(
    checkpoint_setting: Optional[str] = None,
) -> SimpleNamespace:
    args = _base_traffic_args("unimodal")
    args.checkpoint_setting = resolve_traffic_unimodal_checkpoint_setting(
        checkpoint_setting
    )
    return args


def build_traffic_cfa_args(
    checkpoint_setting: Optional[str] = None,
    cfa_bundle_path: Optional[str] = None,
) -> SimpleNamespace:
    args = _base_traffic_args("cfa")
    args.checkpoint_setting = resolve_traffic_unimodal_checkpoint_setting(
        checkpoint_setting
    )
    args.cfa_checkpoint_setting = resolve_traffic_cfa_checkpoint_setting(
        checkpoint_setting
    )
    args.cfa_bundle_path = str(cfa_bundle_path or DEFAULT_TRAFFIC_CFA_BUNDLE)
    args.train_epochs = 10
    args.patience = 5
    return args


def build_domain_args(
    domain: str,
    *,
    mode: str = "cfa",
    checkpoint_setting: Optional[str] = None,
    cfa_bundle_path: Optional[str] = None,
) -> SimpleNamespace:
    from language.apo.timemmd_domains import (
        default_bundle_path,
        get_timemmd_spec,
        normalize_domain,
    )

    domain = normalize_domain(domain)
    if domain in ("agriculture", "agri", "algriculture"):
        if mode == "unimodal":
            return build_agriculture_unimodal_args(checkpoint_setting)
        return build_agriculture_cfa_args(checkpoint_setting, cfa_bundle_path)
    if domain == "traffic":
        if mode == "unimodal":
            return build_traffic_unimodal_args(checkpoint_setting)
        return build_traffic_cfa_args(checkpoint_setting, cfa_bundle_path)

    # Generic Time-MMD domains (climate / economy / …)
    spec = get_timemmd_spec(domain)
    args = _base_monthly_args(
        model_id_prefix=spec.model_id_prefix,
        root_path=str(ROOT / "data" / spec.data_subdir),
        data_path=spec.data_path,
        text_injection_mode=mode,
    )
    args.seasonal_patterns = spec.seasonal
    args.lr_multiplier = spec.lr_multiplier
    args.model_id = (
        f"{spec.model_id_prefix}_2021_8_6_PatchTST_BERT_{mode}_lr{spec.lr_multiplier}"
    )
    setting = build_setting_name(args, 0)
    ckpt = ROOT / "checkpoints" / setting / "checkpoint.pth"
    if mode == "unimodal":
        if checkpoint_setting:
            args.checkpoint_setting = checkpoint_setting
        elif ckpt.exists():
            args.checkpoint_setting = setting
        else:
            args.checkpoint_setting = setting  # expected path after train
        return args

    # CFA
    if checkpoint_setting and "_cfa_" in checkpoint_setting:
        args.checkpoint_setting = checkpoint_setting
        args.cfa_checkpoint_setting = checkpoint_setting
    else:
        args.checkpoint_setting = setting
        args.cfa_checkpoint_setting = setting
    args.cfa_bundle_path = str(
        cfa_bundle_path or default_bundle_path(spec)
    )
    args.train_epochs = 10
    args.patience = 5
    return args


def check_domain_alignment(domain: str) -> AlignmentReport:
    from language.apo.timemmd_domains import csv_path, get_timemmd_spec, normalize_domain

    domain = normalize_domain(domain)
    if domain in ("agriculture", "agri", "algriculture"):
        return assert_domain_aligned(str(AGRI_CSV), max_abs_days=120, min_frac_ok=0.85)
    if domain == "traffic":
        return assert_domain_aligned(
            str(TRAFFIC_CSV), max_abs_days=120, min_frac_ok=0.85
        )
    spec = get_timemmd_spec(domain)
    return assert_domain_aligned(str(csv_path(spec)), max_abs_days=120, min_frac_ok=0.85)


def build_setting_name(args: SimpleNamespace, ii: int = 0) -> str:
    version_suffix = f"vtext_integrated_{args.text_injection_mode}"
    lr_suffix = f"_lr{args.lr_multiplier}x" if args.lr_multiplier != 1.0 else ""
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
        f"ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_"
        f"df{args.d_ff}_expand{args.expand}_dc{args.d_conv}_fc{args.factor}_"
        f"eb{args.embed}_dt{args.distil}_{args.des}_{ii}_llm{args.llm_model}_"
        f"{version_suffix}{lr_suffix}"
    )
