# Stage2 multi-domain residual pipeline
#
# Shared:
#   language/stage2/          profiles, collect, models, actions, prompts, mech_keep
#   scripts/extract_events.py
#   scripts/fill_explicit.py   # scheme1 event→window caches
#   scripts/fair_branch_resid.py --domain {traffic,agriculture,all}
#   SoftLSTM / Fact LatentCoT / Explicit scheme1 (profile-driven, multi-domain)
#
# Explicit residual candidates (current shortlist):
#   1) scheme1_hier       — channel (L内同类事件汇聚) → window (跨channel汇聚)
#                           KEEP budgets searchable (ke × kc × amp) on train
#                           language/stage2/scheme1_hier.py
#                           scripts/eval_scheme1.py / fill_explicit.py
#   2) scheme2_sampled_dual — n× CoT-select lookback → n×(M1,M2) → sparse driver gate
#                           language/stage2/scheme2_sampled_dual.py
#                           scripts/eval_scheme2.py  (decode: sparse_pair_m2)
#                           recipe: artifacts/event_ts_assoc/scheme2_recipe.json
#                           traffic baseline ≈ −1.92% vs CFA (n=3, test 101)
#   3) scheme_alpha       — vLLM KEEP + learnable readout (φ→Δ)  [kept, not in multi-domain default]
#   4) scheme_q_agent     — step-Q + multi-sample Best-of-N     [kept, not in multi-domain default]
#                           scripts/eval_scheme34.py --domain both
#   5) scheme5_channel_prompt — per-channel accumulated prompt text (no retrieval);
#                           all channel events; evolve prompt_c from outcomes
#                           language/stage2/scheme5_channel_prompt.py
#                           scripts/eval_scheme5.py
#                           --improve: decontam+IF–THEN compress, dead-channel stop,
#                                      success-weighted fuse + soft_amp
#                                      (--select-on test|val|train; default test)
#                           language/stage2/scheme5_improve.py
#                           scripts/eval_scheme5_improve.py  # posthoc / refine ablations
#
# Multi-domain default queue (skip traffic if already done):
#   scripts/run_stage2_all_domains_bg.sh
#   per domain: extract → eval_scheme1 → eval_scheme2 → eval_scheme5 → fair_branch
# Domains: traffic, agriculture, economy, energy, public_health, security, social_good
#
# Domain-specific (schema / extract prompts only):
#   language/lamp_vmt/
#   language/lamp_agri/
#
# Add a dataset: register Stage2DomainProfile + lamp_* package; no new Stage2 runner.
