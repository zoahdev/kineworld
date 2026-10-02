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

Training contacts ranged from 132 to 175 per 6,000 transitions. The mean held-out contact fraction was 2.5850% (range 2.4050% to 2.8250%). Across the ten equally sized test sets, 5,170 of 200,000 transitions were contacts.

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

At K=256. Brackets show descriptive seed-level 95% t intervals; all metrics are lower-is-better except signed optimism.

| Model | Ordinary MSE | Contact MSE | Executed cost | Candidate regret | Optimism |
|---|---:|---:|---:|---:|---:|
| Action-blind ridge | 0.0131261 [0.0130122, 0.0132399] | 0.0779683 [0.0736446, 0.0822919] | 0.110021 [0.103006, 0.117036] | 0.0993996 [0.0925658, 0.106233] | 0.0413527 [0.0358172, 0.0468882] |
| Action-conditioned ridge | 0.00345911 [0.00333003, 0.00358819] | 0.102174 [0.096324, 0.108023] | 0.033031 [0.0307337, 0.0353284] | 0.0224097 [0.0201196, 0.0246998] | 0.0222082 [0.0198538, 0.0245626] |
| Quadratic ridge | 0.0034877 [0.0033614, 0.00361401] | 0.103146 [0.0972955, 0.108996] | 0.0337782 [0.0307243, 0.0368321] | 0.0231569 [0.0200351, 0.0262786] | 0.0230016 [0.0197996, 0.0262037] |
| MLP | 0.000665484 [0.000572369, 0.000758599] | 0.0130386 [0.0112444, 0.0148327] | 0.0236217 [0.0210419, 0.0262014] | 0.0130003 [0.0105364, 0.0154642] | 0.0141377 [0.0112804, 0.0169949] |
| Learned hybrid (privileged) | 2.00491e-30 [1.80628e-30, 2.20353e-30] | 2.62417e-31 [1.93772e-31, 3.31063e-31] | 0.0106213 [0.0103598, 0.0108829] | 0 [0, 0] | 8.33284e-17 [5.04636e-17, 1.16193e-16] |
| Same-candidate oracle | 0 [0, 0] | 0 [0, 0] | 0.0106213 [0.0103598, 0.0108829] | 0 [0, 0] | 0 [0, 0] |

The median across-seed ratio of MLP contact MSE to ordinary MSE was 18.716; the median MLP-to-oracle executed-cost ratio was 2.218. These are descriptive post hoc ratios, not independent inferential endpoints.

The hybrid's almost-zero transition error and oracle-matched plan selection serve as a positive implementation control. This is a consequence of correct structural specification plus noiseless observations. It is not evidence of a new learning method. The action-blind model is a negative control; action-conditioned linear regression is the substantive simple control.

| Model | K=1 cost | K=16 cost | K=256 cost | Paired cost change 256 minus 16 |
|---|---:|---:|---:|---:|
| Action-blind ridge | 0.156916 | 0.129706 | 0.110021 | -0.0196849 [-0.0251141, -0.0142558] |
| Action-conditioned ridge | 0.156916 | 0.0544051 | 0.033031 | -0.0213741 [-0.0254544, -0.0172938] |
| Quadratic ridge | 0.156916 | 0.0561913 | 0.0337782 | -0.0224131 [-0.0280973, -0.0167288] |
| MLP | 0.156916 | 0.0400493 | 0.0236217 | -0.0164276 [-0.0197602, -0.0130951] |
| Learned hybrid (privileged) | 0.156916 | 0.0200409 | 0.0106213 | -0.00941952 [-0.00979906, -0.00903999] |
| Same-candidate oracle | 0.156916 | 0.0200409 | 0.0106213 | -0.00941952 [-0.00979906, -0.00903999] |


The expected harmful-budget finding was absent: every mean cost decreased from 16 to 256 candidates. A lower finite-pool oracle cost at the larger budget makes oracle-relative regret the appropriate complementary metric. Both quantities are included rather than equating budget-dependent cost changes with pure model quality.

Post hoc paired comparisons, all in one family of six two-sided tests. Negative differences favor the first-named model or the larger budget. These tests do not upgrade the evidence level.

| Comparison | Paired mean and 95% t interval | Raw p | Bonferroni p |
|---|---:|---:|---:|
| MLP minus ridge: test_mse | -0.00279363 [-0.0029666, -0.00262065] | 4.27188e-11 | 2.56313e-10 |
| MLP minus ridge: actual_cost | -0.00940938 [-0.0124225, -0.00639629] | 5.89244e-05 | 0.000353546 |
| Action-blind ridge: K256 minus K16 cost | -0.0196849 [-0.0251141, -0.0142558] | 1.81246e-05 | 0.000108747 |
| Action-conditioned ridge: K256 minus K16 cost | -0.0213741 [-0.0254544, -0.0172938] | 8.56985e-07 | 5.14191e-06 |
| Quadratic ridge: K256 minus K16 cost | -0.0224131 [-0.0280973, -0.0167288] | 9.187e-06 | 5.5122e-05 |
| MLP: K256 minus K16 cost | -0.0164276 [-0.0197602, -0.0130951] | 1.43473e-06 | 8.6084e-06 |


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
