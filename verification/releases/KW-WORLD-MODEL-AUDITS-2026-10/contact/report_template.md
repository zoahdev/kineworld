# Contact conditional error and planning performance in a synthetic world model

Author: 潘奕成 (Yicheng Pan)  
Date: 2026-10-02  
Status: Reproducibility and diagnostic technical note, limited synthetic evidence (E1), not peer reviewed

## Main finding

The anticipated inversion between ordinary one-step MSE and mean executed-cost rankings, and the harmful planning-budget effect, did not occur. In this fixed synthetic test, a small action-conditioned MLP reduced both held-out prediction error and planning cost relative to ridge regression, and increasing the candidate budget improved every model's mean achieved cost. A simple learned hybrid model given the correct physical structure matched the same-candidate oracle. The experiment therefore supports a narrowly scoped diagnostic, not a new world-model algorithm or a solution to a general world-model research problem.

The remaining MLP error was disproportionately concentrated at wall contacts, and its achieved cost remained above the same-candidate oracle. Reporting event-conditioned errors and directly executed plan costs makes those limitations visible. This finding is specific to the environment, data distribution, function classes, and planner below. It is not evidence that generic neural world models fail, that more planning is harmful, or that hand-specified hybrid models generalize better.

## Motivation and relationship to prior work

Objective mismatch between predictive model fitting and downstream control is established in Lambert et al. (2020). That work directly studies cases in which one-step predictive likelihood does not track control performance. This note does not claim to discover that issue. Chua et al. (2018) established uncertainty-aware ensemble world models and sampling-based planning, while the present experiment deliberately uses a much simpler deterministic open-loop planner and does not reproduce PETS. Yu et al. (2026) advocate decision-centered world-model evaluation, including intervention fidelity, planning, exploitability, and uncertainty. Warrier et al. (2026 version) evaluate environment-level queries beyond observed trajectories. These sources motivate evaluating a learned transition model through decisions, without making this small task a substitute for their benchmarks.

The contribution here is a small auditable artifact: fixed seeds, basic linear and nonlinear controls, a deliberately privileged structural baseline, matched nested candidate pools, event-conditioned metrics, preserved negative results, and deterministic replay. Novelty of the benchmark construction is not asserted.

## Environment and data

The fully observed state is position and velocity, s=(x,v); action a lies in [-1,1]. The free velocity is w=0.9v+0.18a and the proposed position is q=x+w. If q is between 0 and 1 inclusive, the next state is (q,w). If q is outside that interval, the next state is (clip(q,0,1), -0.6w). Equality at a wall is not a new reflection. This clamped-position update is an intentionally simplified hybrid system: it does not integrate the remaining sub-step motion after an impact and should not be described as an accurate rigid-body simulator.

Each training dataset contains 6,000 independent state-action transitions, not trajectories. With probability 0.9, position is uniform on [.25,.75] and velocity on [-.15,.15]; otherwise position is uniform on [0,1] and velocity on [-.5,.5]. Actions are independently uniform on [-1,1]. Validation contains 2,000 transitions and the ordinary held-out test contains 20,000, both from the same mixture. There is no observation noise, partial observability, learned reward, visual encoder, or externally pretrained model.

The ten training seeds are 1000 through 1009. Test, planning, and validation seeds respectively add 1000, 2000, and 3000. Thus the four seed ranges are disjoint. MLP early stopping uses a training-only internal split; external validation selects hyperparameters. Planning starts and goals differ from the transition-test distribution and are described below. Any interpretation of prediction versus planning therefore includes that deliberately specified distribution change.

@@SUPPORT@@

## Models and selection

- Action-blind ridge is a negative control that receives position and velocity but omits action
- Action-conditioned ridge receives position, velocity, and action
- Quadratic ridge expands those inputs to degree-two polynomial features
- MLP uses two 32-unit tanh layers, Adam, standardized inputs, training-only early stopping, and no privileged contact labels
- Learned hybrid uses the correct hybrid functional family and the known arena bounds 0 and 1. It identifies contacts using these known bounds, fits free-motion coefficients by near-unregularized ridge on interior observations, and fits restitution by least squares on contacts. It records observed position extrema but does not discover unknown geometry. Its near-exact recovery is expected in this noiseless correctly specified setting
- Oracle applies the true transition function and is not a trained model

Ridge penalties are selected from {1e-6,1e-3,1}; MLP penalties from {1e-4,1e-2}, using external validation one-step MSE only. No model, environment parameter, planning budget, or selection criterion was changed after the test results were inspected. Full selected penalties and MLP iteration counts are in results.json. These are elementary baselines, not comparisons with current learned video world models or modern model-based reinforcement-learning systems.

## Planning and metrics

For each training seed, 128 independent evaluation starts have x uniform on [.45,.85], v uniform on [-.1,.35], and target g uniform on [.7,.95]. Each episode samples 256 independent eight-action plans from uniform [-1,1]. Budgets K in {1,16,256} use nested prefixes of this same pool, identical across models and oracle. A model rolls out every candidate and selects the least predicted cost. The entire chosen plan is then executed with the true dynamics; there is no replanning or closed-loop feedback.

For horizon H=8, cost is the mean over steps of (x_t-g)^2+0.1v_t^2+0.02a_t^2. States in this cost are the post-action states. Lower cost is better. The oracle chooses the best of the same finite candidate pool; it is not the globally optimal controller. Candidate regret subtracts that same-pool oracle cost. Optimism is actual minus predicted cost for the selected plan and can be negative.

One-step MSE is the average over transitions and both coordinates: half the sum of squared position and velocity errors. Position is normalized to the unit arena; velocity is measured in arena lengths per discrete step. The two coordinates receive equal weight without variance standardization. Contact MSE is the same metric conditional on a true wall-contact transition in the ordinary test set. Noncontact MSE conditions on its complement. These are not independently generated balanced-test results. For contact proportion p, the exact decomposition is overall MSE = p times contact MSE + (1-p) times noncontact MSE.

All confidence intervals are descriptive 95% t intervals over ten independent seed-level means, not intervals treating 1,280 episodes as independent model fits. Actual costs are bounded in this simulator; model-prediction errors and seed summaries are also supplied for inspection. The note makes no multiplicity-adjusted statistical-significance claim. Supplemental comparisons show paired seed-level differences, with Bonferroni-adjusted two-sided t-test p values explicitly labeled post hoc. No thresholded benchmark pass/fail or causal conclusion depends on those tests.

## Results

@@MAIN_TABLE@@

The hybrid's almost-zero transition error and oracle-matched plan selection serve as a positive implementation control. This is a consequence of correct structural specification plus noiseless observations. It is not evidence of a new learning method. The action-blind model is a negative control; action-conditioned linear regression is the substantive simple control.

@@BUDGET_TABLE@@

The expected harmful-budget finding was absent: every mean cost decreased from 16 to 256 candidates. A lower finite-pool oracle cost at the larger budget makes oracle-relative regret the appropriate complementary metric. Both quantities are included rather than equating budget-dependent cost changes with pure model quality.

@@PAIRED_TABLE@@

## Limitations and negative conclusions

This is a single fully observed, noiseless, low-dimensional, hand-designed environment, with ten independent data/model seeds. It provides limited synthetic evidence only. It does not establish external validity for manipulation, visual world models, multi-object contact, partial observability, or real robots. The structural baseline is privileged; it cannot be used as a fair generic-model superiority claim. The oracle is finite-pool. The rankings by ordinary one-step MSE and mean executed cost did not invert. Contact-conditioned MSE can rank models differently and is not interchangeable with ordinary MSE. More candidate search did not damage mean performance over the tested budgets. No null or negative result was removed and no environment tuning was performed to manufacture those phenomena.

The code does not study model uncertainty, ensembles, online data collection, closed-loop MPC, learned rewards, or optimal control. Existing uncertainty-aware and decision-aware methods might close the remaining gap. This experiment neither implements nor compares them. The contact-related observation is diagnostic and does not establish that contact error alone causes all planning regret; isolating that mechanism would require additional preregistered interventions.

## Reproducibility and protocol deviations

The complete locally prespecified protocol is in protocol.md. It was saved before the first run; it was not registered with a third party. The protocol originally mentioned a separate event-balanced probe, but that probe was not executed. Only event-conditioned errors on the ordinary test mixture are reported. Its description of learned geometry was also too broad; the known-wall privilege is corrected explicitly above and in the appended protocol clarification.

After the first run, robustness checks were added to reject confidence intervals with fewer than two seeds, reject non-finite summaries, and produce strict JSON. A complete replay checks that these nonexperimental edits preserve every numeric result. A routing-only absolute output path was replaced by its basename in the public metadata; numeric results were not modified. The manifest records file hashes and these changes. This is an assistant-assisted replay, not external independent replication.

Run from this directory:

    OPENBLAS_NUM_THREADS=1 python run.py --seeds 10 --output results_reproduced.json
    python verify.py --replay results_reproduced.json
    python -m unittest -v test_study.py
    python make_report.py --check

Runtime dependencies and observed package versions are in requirements.txt and results.json. The experiment is CPU-only and requires no network, credentials, user data, GPU, paid service, or pretrained checkpoint. The report's numeric sections are generated by make_report.py from results.json and checked byte-for-byte; verify.py independently reconstructs aggregate metrics and validates plan-selection invariants.

## Authorship and AI assistance

This work is attributed to 潘奕成 (Yicheng Pan) at the author's request. OpenAI's assistant designed and implemented this synthetic diagnostic, executed the experiments and replay, checked the cited sources, and drafted this note. Automated and assistant-based checks are not peer review or independent external validation. No claim is made that the named author personally performed each implementation or verification step. The note uses only the synthetic generator in this package and the cited public literature.

## References

1. Nathan Lambert, Brandon Amos, Omry Yadan, Roberto Calandra. Objective Mismatch in Model-based Reinforcement Learning. L4DC / PMLR 120, 761-770, 2020. https://proceedings.mlr.press/v120/lambert20a.html
2. Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine. Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models. NeurIPS 2018. https://arxiv.org/abs/1805.12114
3. Yang Yu, Shiyuan Zhang, Yifei Sheng, Haoxiang Ren, Haoxin Lin. How Should World Models Be Evaluated for Embodied Decision-Making? A Decision-Making-Centric Position. arXiv:2606.15032v2, June 2026. https://arxiv.org/abs/2606.15032v2
4. Archana Warrier and collaborators. Benchmarking World-Model Learning with Environment-Level Queries. arXiv:2510.19788v4, May 2026. https://arxiv.org/abs/2510.19788v4
