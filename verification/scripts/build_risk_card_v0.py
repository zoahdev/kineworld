"""Build the KineWorld Failure & Risk Card v0 from frozen evidence.

This script does not rerun a model. It validates consistency between the frozen
KW-EXP-0006 manifest and its analysis output, then emits a deterministic,
machine-readable card and a human-readable companion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "verification" / "manifests" / "KW-EXP-0006_manifest.json"
ANALYSIS = ROOT / "results" / "kw_exp_0006_analysis.json"
JSON_OUT = ROOT / "results" / "KW_RISK_CARD_v0.json"
MD_OUT = ROOT / "docs" / "product" / "KINEWORLD_RISK_CARD_v0.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_close(name: str, left: float, right: float, tol: float = 1e-6) -> None:
    if abs(float(left) - float(right)) > tol:
        raise ValueError(f"{name} mismatch: {left} != {right}")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))

    if manifest["experiment_id"] != "KW-EXP-0006" or analysis["experiment"] != "KW-EXP-0006":
        raise ValueError("unexpected source experiment")
    if manifest["n_episodes"] != analysis["n_episodes"]:
        raise ValueError("episode-count mismatch")
    if manifest["n_points"] != analysis["n_points"]:
        raise ValueError("point-count mismatch")
    require_close("median error", manifest["global_median_error"], analysis["global_median"])
    require_close(
        "obj_disp Fisher p",
        manifest["q2_spike_vs_physical"]["obj_disp"]["fisher_p"],
        analysis["q2_spike_vs_physical"]["obj_disp"]["fisher_p"],
    )

    bench = manifest["official_benchmark"]
    obj = manifest["q2_spike_vs_physical"]["obj_disp"]
    horizon = manifest["q1_horizon_curve_median"]
    bands = manifest["q3_physical_ratio_bands"]
    card = {
        "schema": "kineworld.failure-risk-card.v0",
        "card_id": "KW-RISK-CARD-0001",
        "generated_from_frozen_evidence": True,
        "subject": {
            "model_family": "JEPA-WM",
            "task": "Push-T",
            "evaluation_scope": "single checkpoint, seed, and task",
            "training_performed_by_kineworld": False,
        },
        "protocol": {
            "source_experiment": "KW-EXP-0006",
            "episodes": manifest["n_episodes"],
            "horizon_points": manifest["n_points"],
            "cem": manifest["cem"],
            "seed": manifest["seed"],
            "checkpoint_sha256": manifest["artifact_hashes_sha256"]["checkpoint"],
            "evidence_level": "E1 internal reproduction; not independent third-party validation",
        },
        "task_outcome": {
            "success_count": bench["success_count"],
            "episode_count": bench["n_episodes"],
            "success_rate": bench["success_rate"],
            "mean_final_state_distance": bench["actual_state_dist_mean"],
            "median_final_state_distance": bench["actual_state_dist_median"],
        },
        "prediction_risk": {
            "median_latent_error": manifest["global_median_error"],
            "spike_definition": "latent error > Q3 + 3*IQR, frozen before tests",
            "spike_threshold": manifest["spike_threshold"],
            "spike_count": manifest["n_spikes"],
            "spike_rate": manifest["spike_rate"],
            "median_error_by_horizon": horizon,
            "horizon_growth_k1_to_k6": manifest["q1_growth_k1_to_k6_median"],
        },
        "physical_interaction_slice": {
            "condition": "object displacement above the frozen 75th percentile",
            "high_interaction_spike_rate": obj["high_spike_rate"],
            "lower_interaction_spike_rate": obj["low_spike_rate"],
            "relative_risk": round(obj["high_spike_rate"] / obj["low_spike_rate"], 4),
            "fisher_exact_p": obj["fisher_p"],
            "continuous_spearman_rho_including_spikes": manifest["q2_continuous_spearman"]["obj_disp_incl_spike"],
            "continuous_spearman_rho_excluding_spikes": manifest["q2_continuous_spearman"]["obj_disp_excl_spike"],
        },
        "calibration_utility": {
            "physical_ratio_band_width_vs_pooled_at_90pct": bands["alpha_0.1"]["obj_disp"],
            "physical_ratio_band_width_vs_pooled_at_80pct": bands["alpha_0.2"]["obj_disp"],
            "decision": "REJECT physical-ratio bands: statistically real signal did not improve interval efficiency",
        },
        "interpretation": [
            "Aggregate task success does not describe where prediction errors concentrate.",
            "Object motion is associated with higher latent prediction error in this run.",
            "The observed association is diagnostic, not proof of causality or a deployable uncertainty estimator.",
            "High latent error is not equivalent to task failure.",
        ],
        "limitations": manifest["limitations"] + [
            "KineWorld generated this card from its own reproduction; it is not third-party certification.",
            "No comparison against Baize, LeWM, or another model is licensed by this card.",
        ],
        "claims": {
            "allowed": [
                "KineWorld ran a 96-episode JEPA-WM Push-T reproduction and published a hash-linked risk analysis.",
                "In that run, high object displacement was associated with a higher latent-error spike rate.",
                "KineWorld rejected a statistically significant diagnostic when it failed the interval-efficiency gate.",
            ],
            "forbidden": [
                "KineWorld is state of the art or world number one.",
                "KineWorld beats Baize, JEPA-WM, LeWM, or any competitor.",
                "The result is independently or third-party validated.",
                "The physical signal is causal or deployment-ready.",
            ],
        },
        "source_artifacts": {
            "experiment_manifest": {
                "path": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(MANIFEST),
            },
            "analysis": {
                "path": str(ANALYSIS.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(ANALYSIS),
            },
        },
    }
    JSON_OUT.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rr = card["physical_interaction_slice"]["relative_risk"]
    markdown = f"""# KineWorld Failure & Risk Card v0

> Card `KW-RISK-CARD-0001` · E1 internal reproduction · **not third-party certification**

## What was evaluated

One public JEPA-WM Push-T checkpoint was evaluated for **{bench['n_episodes']} episodes** under the frozen KW-EXP-0006 protocol. KineWorld did not train this checkpoint. The purpose of this card is to expose failure structure hidden by an aggregate success rate, not to claim a new model record.

## Outcome and risk signals

| Measure | Observed value | What it supports |
|---|---:|---|
| Task success | **{bench['success_count']}/{bench['n_episodes']} ({bench['success_rate']:.1%})** | Reproduced task outcome for this checkpoint/protocol |
| Latent-error spikes | **{manifest['n_spikes']}/{manifest['n_points']} ({manifest['spike_rate']:.1%})** | Sparse prediction failures exist |
| Median error, horizon 1 → 6 | **{horizon['k=1']:.4f} → {horizon['k=6']:.4f} ({manifest['q1_growth_k1_to_k6_median']:.2f}×)** | Prediction error grows mildly with rollout horizon |
| Spike rate, high object motion | **{obj['high_spike_rate']:.1%}** | Error concentration under stronger physical interaction |
| Spike rate, lower object motion | **{obj['low_spike_rate']:.1%}** | Comparison group under the frozen slice |
| Relative spike risk | **{rr:.2f}×** | Descriptive association; Fisher exact p={obj['fisher_p']:.4f} |

## The important negative result

The physical signal was statistically clear but **not useful as a calibrated risk band**. At nominal 90% coverage, the object-displacement-normalized band was **{bands['alpha_0.1']['obj_disp']:.2f}× wider** than the pooled band; at nominal 80%, it was **{bands['alpha_0.2']['obj_disp']:.2f}× wider**. KineWorld therefore rejected this mechanism instead of presenting significance as product value.

## Reading boundary

This card supports a narrow claim: KineWorld can reproduce a public world-model control run, audit its hidden failure distribution, and reject a diagnostic that fails a predeclared utility gate. It does **not** show that KineWorld beats another model, that object motion causes error, or that this result has independent validation.

## Reproduce and verify

- Machine-readable card: `results/KW_RISK_CARD_v0.json`
- Frozen experiment: `verification/experiments/KW-EXP-0006.md`
- Experiment manifest: `{card['source_artifacts']['experiment_manifest']['path']}` (`{card['source_artifacts']['experiment_manifest']['sha256']}`)
- Analysis artifact: `{card['source_artifacts']['analysis']['path']}` (`{card['source_artifacts']['analysis']['sha256']}`)
- Builder: `verification/scripts/build_risk_card_v0.py`

## Next validation step

An external lab or evaluator should rerun the frozen protocol from the checkpoint hash, compare the generated manifest, and sign its own attestation. Until that happens, the evidence level remains E1.
"""
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text(markdown, encoding="utf-8")
    print(JSON_OUT)
    print(MD_OUT)


if __name__ == "__main__":
    main()
