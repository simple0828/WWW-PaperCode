"""Explicit branch **Candidate 4** — LLM-Agents-with-Q (API / vLLM).

γ-style step actions:
  select_event → select_mech(M1/M2/M3) → optional set_gate / set_amp → abstain/done

vLLM samples N trajectories; local Q trained on residual reward; inference Best-of-N.
LLM weights stay frozen. KEEP budget is raised so denser banks can be used.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from language.lamp_vmt.json_utils import extract_json
from language.stage2.mech_features import (
    pack_keep_features,
    pack_step_features,
    soft_delta_from_selection,
)
from language.stage2.mech_keep import (
    build_mech_bank,
    mse,
    parse_keep_response,
    reward_delta,
)

AMP_TIER = {"off": 0.0, "weak": 0.03, "medium": 0.05, "strong": 0.08, "high": 0.10}

STEP_PROMPT = """You are a residual mechanism agent for {target}.

CFA forecast already exists. Decide step-by-step which EXISTING events/mechanisms
explain why truth may deviate from CFA.

Each step is ONE action:
  - select_event: pick an event key "YYYY-MM/E#" from BANK
  - select_mech: pick M1|M2|M3 under that event (or full id "YYYY-MM/E#/M#")
                 with residual_dir + strength
  - abstain / done: stop adding mechanisms

Rules:
- Prefer covering MULTIPLE months/channels when evidence is rich (not only 1–2 picks).
- KEEP up to {max_keep} mechanisms via select_mech steps. Competing M1/M2/M3 OK.
- Cite ids/events exactly as shown in BANK.
- If evidence weak: gate=0 and keep=[].

BANK ({n_bank} cards):
{bank}

Origin={origin}  CFA={cfa}

Return JSON ONLY:
{{
  "steps":[
    {{"op":"select_event","event":"YYYY-MM/E#","why":"<=10w"}},
    {{"op":"select_mech","id":"YYYY-MM/E#/M1","mech":"M1","residual_dir":"increase|decrease","strength":"weak|medium|strong","why":"<=10w"}},
    {{"op":"select_mech","id":"YYYY-MM/E#/M2","mech":"M2","residual_dir":"increase|decrease","strength":"medium","why":"<=10w"}},
    {{"op":"done"}}
  ],
  "residual_hypothesis":"increase|decrease|flat|mixed",
  "gate":0|1,
  "confidence":0-1,
  "reason":"<=20 words",
  "keep":[{{"id":"...","residual_dir":"increase|decrease","strength":"weak|medium|strong","why":"<=12w"}}]
}}
keep MUST list every successful select_mech (max {max_keep}), or [] if gate=0.
(Amplitude is NOT chosen here; a fixed soft amp + val scale is applied offline.)
"""


def _as_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if hasattr(x, "text"):
        return str(x.text)
    return str(x)


def _norm_event_key(s: str) -> str:
    s = str(s or "").strip()
    # allow YYYY-MM/E0/M1 → YYYY-MM/E0
    if "/M" in s:
        s = s.rsplit("/M", 1)[0]
    return s


def _resolve_mech_id(st: Mapping[str, Any], last_event: str) -> str:
    mid = str(st.get("id") or st.get("mid") or "").strip()
    if mid.count("/") >= 2:
        return mid
    mech = str(st.get("mech") or st.get("m") or "").upper().strip()
    if not mech.startswith("M"):
        # bare 1/2/3
        if mech.isdigit():
            mech = f"M{mech}"
        else:
            mech = ""
    ev = _norm_event_key(str(st.get("event") or last_event or ""))
    if ev and mech:
        return f"{ev}/{mech}"
    return mid


def parse_agent_trajectory(text: str, *, max_keep: int = 8) -> dict[str, Any]:
    """Parse γ-style step agent JSON; synthesize KEEP from select_mech steps."""
    sel = parse_keep_response(text)
    obj: Any = {}
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        obj = extract_json(t) if t else {}
    except Exception:
        try:
            obj = json.loads(t[t.find("{") : t.rfind("}") + 1])
        except Exception:
            obj = {}
    if not isinstance(obj, dict):
        obj = {}

    steps = obj.get("steps") or []
    if not isinstance(steps, list):
        steps = []
    norm_steps = []
    last_event = ""
    keep_from_steps: list[dict[str, Any]] = []
    # amp/gate are NOT step actions; soft amp comes from cfg / val scale offline.
    amp_tier = "medium"
    gate_from_steps = None

    for st in steps[: max_keep + 6]:
        if not isinstance(st, dict):
            continue
        op = str(st.get("op") or st.get("action") or "").lower().strip()
        if op in ("keep", "select"):
            op = "select_mech"
        if op in ("event",):
            op = "select_event"
        if op in ("skip", "drop"):
            op = "abstain"
        if op in ("stop", "end"):
            op = "done"
        # Ignore legacy set_gate / set_amp if model still emits them
        if op in ("gate", "set_gate", "amp", "set_amp", "amplitude"):
            continue
        if op not in ("select_event", "select_mech", "abstain", "done"):
            continue

        entry: dict[str, Any] = {
            "op": op,
            "why": str(st.get("why") or "")[:80],
        }
        if op == "select_event":
            ev = _norm_event_key(str(st.get("event") or st.get("id") or ""))
            entry["event"] = ev
            last_event = ev or last_event
        elif op == "select_mech":
            mid = _resolve_mech_id(st, last_event)
            entry["id"] = mid
            entry["event"] = _norm_event_key(mid)
            entry["mech"] = str(st.get("mech") or (mid.rsplit("/", 1)[-1] if mid else "")).upper()
            entry["residual_dir"] = str(
                st.get("residual_dir") or st.get("dir") or ""
            ).lower()
            entry["strength"] = str(st.get("strength") or "medium").lower()
            if mid:
                last_event = entry["event"] or last_event
                keep_from_steps.append(
                    {
                        "id": mid,
                        "residual_dir": entry["residual_dir"] or "increase",
                        "strength": entry["strength"],
                        "why": entry["why"],
                    }
                )
        norm_steps.append(entry)

    # Prefer keep from steps when present; else LLM keep list (capped).
    if keep_from_steps:
        sel["keep"] = keep_from_steps[:max_keep]
    else:
        sel["keep"] = (sel.get("keep") or [])[:max_keep]

    if gate_from_steps is not None:
        sel["gate"] = 1 if gate_from_steps == 1 else 0
    elif obj.get("gate") is not None:
        try:
            sel["gate"] = 1 if int(obj.get("gate")) == 1 else 0
        except Exception:
            pass
    elif sel.get("keep") and int(sel.get("gate") or 0) == 0 and obj.get("gate") is None:
        sel["gate"] = 1

    # soft amp fixed; val-scale applied in attach_scheme_q
    sel["amp"] = "medium"
    sel["soft_amp"] = float(AMP_TIER["medium"])
    sel["steps"] = norm_steps
    sel["n_select_event"] = sum(1 for s in norm_steps if s["op"] == "select_event")
    sel["n_select_mech"] = sum(1 for s in norm_steps if s["op"] == "select_mech")
    return sel


@dataclass
class SchemeQConfig:
    n_samples: int = 3
    soft_amp: float = 0.05  # default if trajectory omits set_amp
    temperature: float = 0.85
    max_tokens: int = 700
    ridge_alpha: float = 1.0
    use_step_q: bool = True
    max_keep: int = 8
    max_per_month: int = 6
    max_mechs_per_event: int = 3
    cache_tag: str = "v3"  # select_event/select_mech only; amp via val scale
    scale_grid: tuple[float, ...] = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5)


def _render(template: str, *, target: str, origin: str, cfa, bank, max_keep: int) -> str:
    bank_txt = "\n".join(c.as_prompt_line() for c in bank) or "  (empty)"
    return template.format(
        target=target,
        origin=origin,
        cfa=np.round(np.asarray(cfa, float), 4).tolist(),
        bank=bank_txt,
        max_keep=max_keep,
        n_bank=len(bank),
    )


def _load_multi_cache(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not path.is_file():
        return out
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        wid = row["window_id"]
        idx = int(row.get("sample_idx") or 0)
        bucket = out.setdefault(wid, [])
        while len(bucket) <= idx:
            bucket.append("")
        bucket[idx] = row["response"]
    return out


def _append_multi_cache(path: Path, window_id: str, sample_idx: int, response: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "window_id": window_id,
                    "sample_idx": sample_idx,
                    "response": response,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def ensure_multi_keep(
    packs: Sequence[dict[str, Any]],
    *,
    profile,
    client,
    cache_path: Path,
    n_samples: int,
    fill_missing: bool,
    temperature: float,
    max_tokens: int,
    max_keep: int,
    wids: Optional[set[str]] = None,
) -> dict[str, list[str]]:
    bucket = _load_multi_cache(cache_path)
    need: list[tuple[str, int, str]] = []
    wid_set = wids or {p["row"]["wid"] for p in packs}
    for p in packs:
        wid = p["row"]["wid"]
        if wid not in wid_set:
            continue
        cur = bucket.get(wid) or []
        prompt = _render(
            STEP_PROMPT,
            target=profile.target_series_short,
            origin=p["row"].get("origin") or "",
            cfa=p["row"]["yc"],
            bank=p["bank"],
            max_keep=max_keep,
        )
        for i in range(n_samples):
            if i < len(cur) and cur[i]:
                continue
            need.append((wid, i, prompt))
    if need and fill_missing and client is not None:
        sys = (
            "You are a step-level residual mechanism agent. "
            "Use select_event then select_mech(M1/M2/M3). "
            "Do not choose amplitude. JSON only. No markdown."
        )
        chunk = 12
        for s in range(0, len(need), chunk):
            part = need[s : s + chunk]
            print(f"  [q-agent] LLM samples {s}/{len(need)}", flush=True)
            resps = client.ask_many(
                [x[2] for x in part],
                system=sys,
                temperature=temperature,
                max_tokens=max_tokens,
                concurrency=min(12, len(part)),
            )
            for (wid, idx, _), text in zip(part, resps):
                text = _as_text(text)
                lst = bucket.setdefault(wid, [])
                while len(lst) <= idx:
                    lst.append("")
                lst[idx] = text
                _append_multi_cache(cache_path, wid, idx, text)
    return bucket


def attach_scheme_q(
    rows: list[dict[str, Any]],
    by_ym: Mapping[str, Sequence[dict]],
    profile,
    H: int,
    *,
    client=None,
    cache_dir: Optional[Path] = None,
    train_wids: Optional[set[str]] = None,
    val_wids: Optional[set[str]] = None,
    channels: Sequence[str] = (),
    cfg: Optional[SchemeQConfig] = None,
    fill_missing: bool = True,
) -> dict[str, Any]:
    cfg = cfg or SchemeQConfig()
    train_set = train_wids or {r["wid"] for r in rows if str(r["wid"]).startswith("train:")}
    val_set = val_wids or set()
    focus = profile.focus
    chans = tuple(channels) if channels else tuple(focus)

    packs = []
    for r in rows:
        packs.append(
            {
                "row": r,
                "bank": build_mech_bank(
                    by_ym,
                    r.get("months") or [],
                    focus,
                    max_per_month=cfg.max_per_month,
                    max_mechs_per_event=cfg.max_mechs_per_event,
                ),
            }
        )

    cache_dir = Path(cache_dir) if cache_dir else Path("/tmp/scheme_q")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"q_{cfg.cache_tag}_keep{cfg.max_keep}_n{cfg.n_samples}.jsonl"

    bucket = ensure_multi_keep(
        packs,
        profile=profile,
        client=client,
        cache_path=cache_path,
        n_samples=cfg.n_samples,
        fill_missing=fill_missing,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        max_keep=cfg.max_keep,
    )

    X_traj, y_traj = [], []
    X_step, y_step = [], []

    def _traj_amp(sel: Mapping[str, Any]) -> float:
        if "soft_amp" in sel:
            return float(sel["soft_amp"])
        return float(cfg.soft_amp)

    def _traj_reward(row, bank, sel) -> float:
        d = soft_delta_from_selection(bank, sel, H, _traj_amp(sel))
        return reward_delta(row["yc"], row["yt"], d)

    bank_sizes = []
    keep_counts = []
    for p in packs:
        wid = p["row"]["wid"]
        bank_sizes.append(len(p["bank"]))
        if wid not in train_set:
            continue
        samples = [s for s in (bucket.get(wid) or []) if s][: cfg.n_samples]
        for text in samples:
            sel = parse_agent_trajectory(text, max_keep=cfg.max_keep)
            keep_counts.append(len(sel.get("keep") or []))
            rwd = _traj_reward(p["row"], p["bank"], sel)
            X_traj.append(
                pack_keep_features(
                    p["bank"],
                    sel,
                    p["row"]["yc"],
                    H,
                    channels=chans,
                    soft_amp=_traj_amp(sel),
                )
            )
            y_traj.append(rwd)
            if cfg.use_step_q and sel.get("steps"):
                partial: list[dict] = []
                for st in sel["steps"]:
                    X_step.append(
                        pack_step_features(
                            p["bank"],
                            partial,
                            st,
                            p["row"]["yc"],
                            H,
                            channels=chans,
                        )
                    )
                    y_step.append(rwd)
                    if st.get("op") == "select_mech" and st.get("id"):
                        partial.append(
                            {
                                "id": st["id"],
                                "residual_dir": st.get("residual_dir"),
                                "strength": st.get("strength"),
                                "why": st.get("why"),
                            }
                        )

    def _fit(Xs, ys):
        if len(Xs) < 8:
            return None
        X = np.stack(Xs).astype(np.float32)
        y = np.asarray(ys, float)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=cfg.ridge_alpha)),
            ]
        )
        pipe.fit(X, y)
        return pipe

    q_traj = _fit(X_traj, y_traj)
    q_step = _fit(X_step, y_step) if cfg.use_step_q else None

    def score_sample(p, sel) -> float:
        scores = []
        if q_traj is not None:
            f = pack_keep_features(
                p["bank"],
                sel,
                p["row"]["yc"],
                H,
                channels=chans,
                soft_amp=_traj_amp(sel),
            )
            scores.append(float(q_traj.predict(f.reshape(1, -1))[0]))
        if q_step is not None and sel.get("steps"):
            partial = []
            step_scores = []
            for st in sel["steps"]:
                f = pack_step_features(
                    p["bank"], partial, st, p["row"]["yc"], H, channels=chans
                )
                step_scores.append(float(q_step.predict(f.reshape(1, -1))[0]))
                if st.get("op") == "select_mech" and st.get("id"):
                    partial.append(
                        {
                            "id": st["id"],
                            "residual_dir": st.get("residual_dir"),
                            "strength": st.get("strength"),
                        }
                    )
            if step_scores:
                scores.append(float(np.mean(step_scores)))
        if not scores:
            return float(sel.get("conf") or 0) * float(int(sel.get("gate") or 0)) * (
                0.15 + 0.1 * len(sel.get("keep") or [])
            )
        return float(np.mean(scores))

    def best_of_n_delta(p) -> tuple[np.ndarray, dict]:
        samples = [s for s in (bucket.get(p["row"]["wid"]) or []) if s][: cfg.n_samples]
        if not samples:
            z = np.zeros(H, np.float32)
            return z, {"n": 0, "picked": -1, "q": 0.0}
        best_i, best_q, best_sel = 0, -1e18, parse_agent_trajectory(
            samples[0], max_keep=cfg.max_keep
        )
        for i, text in enumerate(samples):
            sel = parse_agent_trajectory(text, max_keep=cfg.max_keep)
            q = score_sample(p, sel)
            if q > best_q:
                best_q, best_i, best_sel = q, i, sel
        d = soft_delta_from_selection(p["bank"], best_sel, H, _traj_amp(best_sel))
        return d, {
            "n": len(samples),
            "picked": best_i,
            "q": best_q,
            "gate": best_sel.get("gate"),
            "amp": best_sel.get("amp"),
            "n_keep": len(best_sel.get("keep") or []),
            "n_select_event": best_sel.get("n_select_event"),
            "n_select_mech": best_sel.get("n_select_mech"),
            "steps": len(best_sel.get("steps") or []),
            "bank_size": len(p["bank"]),
        }

    val_ids = [p["row"]["wid"] for p in packs if p["row"]["wid"] in val_set]
    if not val_ids:
        val_ids = [p["row"]["wid"] for p in packs if p["row"]["wid"] in train_set][::5]
    by_wid = {p["row"]["wid"]: p for p in packs}
    best_scale, best_vm = 1.0, 1e18
    for s in cfg.scale_grid:
        ms = []
        for wid in val_ids:
            p = by_wid[wid]
            d, _ = best_of_n_delta(p)
            ms.append(mse(p["row"]["yc"] + d * s, p["row"]["yt"]))
        vm = float(np.mean(ms)) if ms else 1e18
        if vm < best_vm:
            best_vm, best_scale = vm, float(s)

    n_q_pick = 0
    keep_test = []
    for p in packs:
        d, meta = best_of_n_delta(p)
        p["row"]["mech_q"] = (d * best_scale).astype(np.float32)
        p["row"]["mech_q_unscaled"] = d.astype(np.float32)
        p["row"]["mech_q_meta"] = {**meta, "scale": best_scale}
        if meta.get("n", 0) > 1:
            n_q_pick += 1
        if str(p["row"]["wid"]).startswith("test:"):
            keep_test.append(meta.get("n_keep") or 0)

    return {
        "stats": {
            "version": cfg.cache_tag,
            "n_samples": cfg.n_samples,
            "max_keep": cfg.max_keep,
            "max_per_month": cfg.max_per_month,
            "n_traj_train": len(X_traj),
            "n_step_train": len(X_step),
            "q_traj_fit": q_traj is not None,
            "q_step_fit": q_step is not None,
            "best_scale": best_scale,
            "best_val_mse": best_vm,
            "windows_with_multi": n_q_pick,
            "bank_size_mean": float(np.mean(bank_sizes)) if bank_sizes else 0.0,
            "train_keep_mean": float(np.mean(keep_counts)) if keep_counts else 0.0,
            "test_keep_mean": float(np.mean(keep_test)) if keep_test else 0.0,
            "cache": str(cache_path),
        }
    }
