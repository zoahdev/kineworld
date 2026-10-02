# Frozen evaluation protocol (2026-10-02)

A small, original synthetic environment for auditing learned action-conditioned dynamics; no external models or private code/data.

State s=(x,v), control a in [-1,1]. Free velocity w=0.9 v + 0.18 a; candidate position q=x+w. If q>1, x'=1, v'=-0.6 w; if q<0, x'=0, v'=-0.6 w; otherwise (x',v')=(q,w). Position clamping and restitution deliberately make this a simplified hybrid system rather than an accurate rigid-body simulator.

Training: independently sampled state-action triples, 90% x~U(.25,.75), v~U(-.15,.15), a~U(-1,1), 10% x~U(0,1), v~U(-.5,.5), same actions. n=6000. Validation n=2000, same mixture; test n=20000, same mixture. Independent seed ranges. Separate event-balanced probe is diagnostic, not the model-selection metric.

Models: action-blind ridge; action-conditioned ridge; degree-two polynomial ridge; MLP with (32,32) tanh layers and standardization; a learned hybrid structural baseline fitted only from transition observations. Model hyperparameters selected by validation one-step MSE where applicable, no test selection. Hybrid baseline is explicitly privileged by the correct functional family but learns coefficients/walls/restitution from training observations. Oracle is a planner ceiling, not a learned comparator.

Planning: open-loop H=8 candidate action sequences, i.i.d. U(-1,1), budgets K={1,16,256}. Nested candidates and identical candidates across models/oracle. Test initial x~U(.45,.85), v~U(-.1,.35), goal~U(.7,.95). Objective is sum_t [(x_t-goal)^2+.1 v_t^2] + .02 sum_t a_t^2, divided by H. Plan chosen entirely in learned rollout, executed in true simulator. Oracle chooses best of the same candidates. Report actual cost, candidate-oracle regret, and predicted-vs-actual cost gap. All models get the same objective and planner. This is open-loop model-predictive planning evaluation, not closed-loop MPC.

Confirmatory run: 10 independently trained models per type, seeds 1000..1009. Per training seed, 128 held-out planning episodes with disjoint seed 3000+i; test transitions seed 2000+i and validation 4000+i. Dataset dimensions, model choices, budgets and environment remain fixed. Intervals: 95% t intervals over 10 seed-level means; paired differences for requested comparisons. Episode-level pseudo-replication not used. We will report null and negative results, not alter the environment to make an expected ranking occur.

This document is a local prespecification, not a registered preregistration. Main purpose is a reproducibility/diagnostic note. The existing objective-mismatch and model-exploitation literatures preclude claiming those phenomena as new.

## Execution clarifications recorded after the first run

- The implementation reports contact-conditioned and noncontact-conditioned errors on the ordinary held-out mixture. The separately described event-balanced probe was not executed. No independent balanced-probe result is claimed.
- The hybrid model uses the known arena bounds 0 and 1 to identify training contacts. It is privileged by both these bounds and the correct hybrid functional family. It fits free-motion coefficients and restitution; its observed extrema are not evidence that it discovered unknown geometry.
- Reported confidence intervals require at least two seed replicates. A post-run robustness correction rejects smaller CLI runs and non-finite summaries, and writes strict JSON. This does not change any recorded ten-seed experimental setting or number. A complete deterministic replay checks this.
