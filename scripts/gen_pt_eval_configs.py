# -*- coding: utf-8 -*-
"""
Generate Push-T planning-eval configs for jepa-wms pretrained checkpoints.

Replicates app/vjepa_wm/utils.py::build_plan_eval_args() merge logic faithfully
(no imports from the repo; pure YAML surgery), so the dumped configs are
byte-equivalent in semantics to what `plan_only_eval_mode + dump_eval_configs`
would produce.

Outputs:
  external/jepa-wms/configs/dump_online_evals/pt/<name>.yaml   (repo-standard location)
  experiments/manifests/<name>.manifest.json                    (provenance manifest)

Usage:
  python gen_pt_eval_configs.py --planner cem --cost L2 --episodes 96
  python gen_pt_eval_configs.py --planner cem --cost L2 --episodes 1 --suffix smoke --quick-debug
"""
import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "jepa-wms"))
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TRAIN_CFG = os.path.join(
    REPO, "configs", "vjepa_wm", "pt_sweep",
    "pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save.yaml",
)
TEMPLATE_DIR = os.path.join(REPO, "configs", "online_plan_evals", "pt")
DUMP_DIR = os.path.join(REPO, "configs", "dump_online_evals", "pt")
CKPT = os.path.join(WS, "checkpoints", "jepa_wm_pusht.pth.tar")
CKPT_DIR = os.path.join(WS, "checkpoints")
RESULTS_ROOT = os.path.join(WS, "results", "phase0_jepa_wm_pusht")

MODULE_NAME = "app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds"


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="cem", choices=["cem", "gd", "ng", "adam"])
    ap.add_argument("--cost", default="L2", choices=["L1", "L2"])
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--quick-debug", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    # Reduced-CEM budget overrides (Phase-0 benchmark strategy, recorded in manifest).
    # When omitted the planner inherits the template defaults (official 30x300x10).
    ap.add_argument("--cem_iter", type=int, default=None, help="CEM iterations (official=30)")
    ap.add_argument("--cem_samples", type=int, default=None, help="CEM num_samples (official=300)")
    ap.add_argument("--cem_elites", type=int, default=None, help="CEM num_elites (official=10)")
    # Closed-loop replanning override (official Push-T=6 = fully open-loop per episode).
    # num_act_stepped < horizon makes the official evaluator replan mid-episode
    # (while-loop in plan_evaluator.py), e.g. 2 => replan every 2 latent steps.
    ap.add_argument("--num_act_stepped", type=int, default=None, help="actions stepped per plan (official pt=6)")
    args = ap.parse_args()

    train = load_yaml(TRAIN_CFG)
    cfgs_model = train["model"]
    cfgs_data = train["data"]
    cfgs_data_aug = train["data_aug"]
    cfgs_plan_evals = train["evals"]

    # template path mirrors repo convention (cem/gd/ng/adam subdirs)
    if args.planner == "cem":
        tpl_path = os.path.join(TEMPLATE_DIR, f"pt_{args.cost}_cem_sourcedset_H6_nas6_ctxt2.yaml")
    else:
        tpl_path = os.path.join(TEMPLATE_DIR, args.planner, f"pt_{args.cost}_{args.planner}_sourcedset_H6_nas6_ctxt2.yaml")
    planning_cfg = load_yaml(tpl_path)

    # --- replicate build_plan_eval_args merge ---
    model_kwargs = planning_cfg.get("model_kwargs", {})
    model_kwargs["module_name"] = MODULE_NAME
    model_kwargs["checkpoint"] = CKPT  # absolute path: supported per vit_enc_preds.init_module
    model_kwargs["pretrain_kwargs"].update(cfgs_model)
    model_kwargs["data"] = cfgs_data  # override_cfgs_data=True in training config
    model_kwargs["data"]["img_size"] = cfgs_data.get("img_size", 256)
    model_kwargs["data_aug"] = cfgs_data_aug
    planning_cfg["model_kwargs"] = model_kwargs

    planning_cfg["folder"] = RESULTS_ROOT
    planning_cfg["checkpoint_folder"] = CKPT_DIR

    planning_cfg["task_specification"]["num_frames"] = cfgs_model["tubelet_size_enc"]
    planning_cfg["task_specification"]["num_proprios"] = cfgs_model["tubelet_size_enc"]
    if "img_size" in cfgs_data:
        planning_cfg["task_specification"]["img_size"] = cfgs_data["img_size"]
    # evals.* overrides from training config
    planning_cfg["planner"]["decode_each_iteration"] = cfgs_plan_evals.get("decode", True)
    evals_obs = cfgs_plan_evals.get("obs")
    if evals_obs is not None:
        planning_cfg["task_specification"]["obs"] = evals_obs
    evals_alpha = cfgs_plan_evals.get("alpha")
    if evals_alpha is not None:
        planning_cfg["planner"]["planning_objective"]["alpha"] = evals_alpha

    # Reduced-CEM budget overrides (Phase-0 strategy). Keep official values in meta
    # for audit, then overwrite planner budgets with the reduced ones.
    official_cem = {
        "iterations": planning_cfg["planner"].get("iterations"),
        "num_samples": planning_cfg["planner"].get("num_samples"),
        "num_elites": planning_cfg["planner"].get("num_elites"),
    }
    if args.cem_iter is not None:
        planning_cfg["meta"]["official_cem_iterations"] = official_cem["iterations"]
        planning_cfg["planner"]["iterations"] = args.cem_iter
    if args.cem_samples is not None:
        planning_cfg["meta"]["official_cem_num_samples"] = official_cem["num_samples"]
        planning_cfg["planner"]["num_samples"] = args.cem_samples
    if args.cem_elites is not None:
        planning_cfg["meta"]["official_cem_num_elites"] = official_cem["num_elites"]
        planning_cfg["planner"]["num_elites"] = args.cem_elites
    if args.num_act_stepped is not None:
        planning_cfg["meta"]["official_num_act_stepped"] = planning_cfg["planner"].get("num_act_stepped")
        planning_cfg["planner"]["num_act_stepped"] = args.num_act_stepped

    # eval-run specific overrides (our choices, documented)
    planning_cfg["meta"]["eval_episodes"] = args.episodes
    planning_cfg["meta"]["seed"] = args.seed
    if args.quick_debug:
        planning_cfg["meta"]["quick_debug"] = True

    name = f"pt_{args.cost}_{args.planner}_sourcedset_H6_nas6_ctxt2_ep{args.episodes}"
    if args.suffix:
        name += f"_{args.suffix}"
    planning_cfg["tag"] = f"phase0/{name}"

    os.makedirs(DUMP_DIR, exist_ok=True)
    out_path = os.path.join(DUMP_DIR, f"{name}.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(planning_cfg, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    # provenance manifest
    os.makedirs(os.path.join(WS, "experiments", "manifests"), exist_ok=True)
    manifest = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "generator": os.path.basename(__file__),
        "template": os.path.relpath(tpl_path, REPO),
        "training_config": os.path.relpath(TRAIN_CFG, REPO),
        "merge_logic": "app/vjepa_wm/utils.py::build_plan_eval_args (override_cfgs_data=True)",
        "checkpoint": CKPT,
        "checkpoint_bytes": os.path.getsize(CKPT) if os.path.exists(CKPT) else None,
        "overrides": {
            "eval_episodes": args.episodes,
            "seed": args.seed,
            "quick_debug": args.quick_debug,
            "planner": args.planner,
            "cost": args.cost,
            "obs": planning_cfg["task_specification"].get("obs"),
            "alpha": planning_cfg["planner"]["planning_objective"]["alpha"],
            "cem_iterations": planning_cfg["planner"].get("iterations"),
            "cem_num_samples": planning_cfg["planner"].get("num_samples"),
            "cem_num_elites": planning_cfg["planner"].get("num_elites"),
            "num_act_stepped": planning_cfg["planner"].get("num_act_stepped"),
        },
        "out_path": out_path,
    }
    with open(os.path.join(WS, "experiments", "manifests", f"{name}.manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[OK] {out_path}")
    print(f"     planner={args.planner} cost={args.cost} episodes={args.episodes} "
          f"obs={planning_cfg['task_specification']['obs']} alpha={planning_cfg['planner']['planning_objective']['alpha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
