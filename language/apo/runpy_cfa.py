"""Export CFA weights from official run.py Exp (PatchTST + fusion MLP)."""

from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from language.apo.config import build_setting_name


def resolve_cfa_checkpoint_setting(args: SimpleNamespace) -> str:
    """Setting folder name for CFA run.py training (distinct from unimodal ckpt)."""
    override = getattr(args, "cfa_checkpoint_setting", None)
    if override:
        return str(override)
    cfa_args = deepcopy(args)
    cfa_args.text_injection_mode = "cfa"
    return build_setting_name(cfa_args, 0)


def cfa_bundle_in_checkpoint_dir(args: SimpleNamespace) -> Path:
    setting = resolve_cfa_checkpoint_setting(args)
    return Path(args.checkpoints) / setting / "cfa_bundle.pt"


def _ensure_runpy_train_fields(args: SimpleNamespace) -> SimpleNamespace:
    """Add run.py training fields missing from APO config namespaces."""
    out = deepcopy(args)
    defaults = {
        "train_epochs": 10,
        "patience": 5,
        "itr": 1,
        "is_training": 1,
    }
    for key, val in defaults.items():
        if not hasattr(out, key):
            setattr(out, key, val)
    return out


def export_runpy_cfa_bundle(
    args,
    dest: str | Path,
    *,
    force_retrain: bool = False,
    run_test: bool = False,
) -> Path:
    """Ensure run.py CFA bundle exists and copy to dest for APO.

    run.py saves PatchTST in checkpoint.pth; fusion MLP lives only in cfa_bundle.pt
    (written at end of Exp.train). If cfa_bundle.pt is missing, re-runs Exp.train.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    train_args = _ensure_runpy_train_fields(args)
    train_args.text_injection_mode = "cfa"
    train_args.text_emb = int(getattr(train_args, "pred_len", train_args.text_emb))
    setting = resolve_cfa_checkpoint_setting(train_args)
    ckpt_dir = Path(train_args.checkpoints) / setting
    src = ckpt_dir / "cfa_bundle.pt"

    if force_retrain or not src.exists():
        if ckpt_dir.joinpath("checkpoint.pth").exists() and not src.exists():
            print(
                f"Found {ckpt_dir}/checkpoint.pth but no cfa_bundle.pt; "
                "re-running Exp.train to export MLP+PatchTST"
            )
        from exp.exp_long_term_forecasting_text_integrated import Exp_Long_Term_Forecast

        exp = Exp_Long_Term_Forecast(train_args)
        print(f">>>>>>> run.py CFA train/export: {setting}")
        exp.train(setting)
        if run_test:
            exp.test(setting, test=1)
        if not src.exists():
            raise FileNotFoundError(
                f"Expected CFA bundle after train: {src}"
            )

    shutil.copy2(src, dest)
    print(f"Exported run.py CFA bundle → {dest}")
    return dest
