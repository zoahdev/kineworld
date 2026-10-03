# Learned-model window certification: independent audit and a precise repair

**Author:** 潘奕成 (Yicheng Pan)  
**Version:** 1.0, 2026-10-03  
**Status:** Public technical report. Not peer-reviewed. Mathematical arguments and bounded synthetic checks; no real-world control improvement is established.

**AI assistance:** OpenAI AI tools substantially assisted with derivations, literature checks, implementation, executable testing, and writing. Separate automated checks and reruns are internal AI-assisted verification, not external human peer review. No human expert endorsement, institutional affiliation, or independent institutional replication is claimed.

## Bottom line

The original theorem in [the corrected finite-family baseline](FINITE_FAMILY_BASELINE.md) is sound in its expressly stated **fixed finite, realizable candidate-family** regime, with the causal-likelihood and support clarifications below. It does not establish a learned neural-ensemble guarantee. A two-member same-data fitted ensemble can exclude the truth with probability one.

A rigorously valid extension uses an online predictor whose probability for each outcome is committed before that outcome arrives, and inverts its likelihood-ratio test over an entire predeclared, possibly continuous, realizable parameter class. This also permits unknown hidden initial priors. It is an application of established sequential inference, not a new statistical principle.

The extension does work on a bounded genuinely state-changing, partially observed model with one unknown continuous transition parameter: all five predeclared runs selected memory 2, with per-trajectory 95%-level theorem bounds approximately 0.508–0.517 at discount 0.5. The underlying reward range is [0,1], so the trivial discounted loss bound is 2. These are certification results, not measured policy improvements.

The main unresolved issue is now explicit: a global, all-histories closure target can remain statistically unidentifiable from one irreversible trajectory. Moreover, an arbitrary-hidden-prior column envelope is not invariant to unreachable latent-state augmentation, so it can be vacuous even when the observable process has memory zero. Removing unreachable states repairs that particular representational artifact; rare reachable modes remain a substantive statistical obstacle.

## 1. Audit of the original statistical claim

### 1.1 Same-data fitting and selection

Valid: choose a global memory length from a fixed finite menu after seeing all data; choose a stopping time; estimate the empirical transitions using the same observations as the model confidence set. Both probability events are simultaneous, so their intersection needs no independence.

Invalid: train an arbitrary collection of kernels on the trajectory, call the resulting collection a finite family, and substitute the realized collection size in the fixed-family likelihood cutoff. The original written theorem explicitly disallows this, but its result must not be described as unrestricted learned-model certification.

Exact counterexample: observe T independent fair bits. Include the true fair-coin process plus a deterministic finite-state model fitted to emit exactly the observed T-bit string (then emit zeros). On every possible dataset, fitted likelihood is 1 and true likelihood is 2^(-T). At T=6 and delta_model=0.05, the naive size-2 cutoff is 40, while the ratio is 64. Truth is excluded on every dataset. The full predeclared family contains all 64 possible memorizing models plus truth; its valid size-65 cutoff does not exclude truth this way.

This is not fixed merely by a small parameter dimension: if a two-member list consists of Bernoulli(1/2) and the current empirical Bernoulli MLE at every time, the generalized log-likelihood ratio is unbounded almost surely by the law of the iterated logarithm. A constant size-2 cutoff is therefore not an anytime guarantee.

### 1.2 Adaptive actions

The appropriate likelihood is the **causal observation likelihood**: multiply each candidate's next-observation probability along the executed action sequence, integrating out hidden states. Equivalently, take the joint trajectory likelihood under the same observable-history policy and cancel its action factors in a ratio.

It is not the ordinary conditional probability of all observations given the entire realized adaptive action vector. Future actions can disclose earlier observations. For example, if A1=O1 deterministically, conditioning on A1 reveals O1, whereas the causal initial-observation likelihood still includes its probability.

The policy must use only the observed past and independent external randomization. The proof does not cover an action rule secretly using hidden states or future observations.

### 1.3 Hidden initial priors

The original finite-family theorem needs the actual initial prior among its declared candidates. Taking an arbitrary-prior supremum in the window envelope does not retrospectively fix an incorrectly specified likelihood.

Unknown initial priors can be included as nuisance parameters and projected out with the repaired construction in Section 2. The example in Section 4 does not need their identity because its test factors use only transitions starting from an observation that reveals the current bit.

### 1.4 Predictable visit selection and all-time/all-window control

For each predeclared context c=(m,w,a), the inclusion indicator is known after A_t and before O_(t+1). This is sufficient for the visit-time Hoeffding supermartingale, even with overlapping windows, adaptive actions, nonstationarity, and no resets.

The empirical target is the average of the actual full-history conditional laws at the selected visits. It is not generally a fixed population transition, nor an unconditional expectation of a random-ratio estimator.

A union over the J fixed contexts, all 2^d observation subsets, and positive visit counts n, with weights 6/(pi^2 n^2), gives exactly the original radius

b(n)=min{1, sqrt(log(pi^2 J 2^d n^2/(6 delta_data))/(2n))}, b(0)=1.

The bound remains valid at arbitrary calendar times because counts change only at visits. Data-selected global m is covered. A learned encoder that retrospectively reassigns visits is not covered just because each final label is finite-valued; its future-dependent grouping destroys this argument unless a predeclared class union or separate evaluation scheme justifies it.

### 1.5 Support, startup, and numerical exactness

State prediction claims apply to **true-supported causal histories**. Conditional distributions at impossible histories are undefined. For a candidate/window, discard exactly zero-likelihood columns; never discard small positive columns merely for convenience. For example, A=diag(1,epsilon), epsilon>0, has normalized columns at TV distance 1 for every epsilon, although a positivity tolerance may drop the second column and report zero.

If no candidate/window remains, abstain. A candidate that cannot realize a particular window contributes no conditional laws there. One may plan on a verified support-closed state set; unvisited but possible states still need radius 1. Empty confidence sets are not a certificate of low error.

Only full windows are pooled. A fresh short prefix generated from the initial prior has a different target from a recurring short suffix. The theorem correctly excludes startup histories shorter than m. It selects a frozen global m and a recursive shift state; arbitrary per-step changes of context length are not justified.

### 1.6 Misspecification

Neither likelihood confidence nor abundant visits verifies realizability. A singleton confidence family is always nonempty, even if wrong.

Concrete failure if realizability is removed: the candidate is iid Bernoulli(beta), while truth first chooses a permanent hidden bit, then emits iid Bernoulli(beta) or Bernoulli(1-beta), 0<beta<1/2. In the beta branch, every finite window receives infinitely many visits, empirical next-observation frequencies approach beta, and the singleton envelope is zero. Nevertheless the true process has supported histories with the same fixed suffix and arbitrarily different old-prefix posteriors, producing next-observation probabilities close to both beta and 1-beta. The claimed all-histories prediction guarantee fails with positive probability. No empty confidence set warns the learner.

### 1.7 Coverage corollary and planning

The original fresh-block coverage argument is valid when actions are conditionally uniform and all relevant observation conditionals, including the initial one when used, exceed beta. A prescribed m-observation window and its next action has conditional block probability at least (beta/|A|)^m. Adapted Bernoulli lower-tail concentration needs no independence or latent mixing. This addresses visitation only, not closure or model identification.

The discounted planning factor 2 gamma/(1-gamma)^2 is correct for half-L1 total variation and known [0,1] observation/action reward. It follows from two Bellman comparison bounds. The learned empirical model and selected global window must be frozen for the evaluated policy. The result does not automatically cover repeatedly replanning into a changing representation.

## 2. Repaired theorem for learned predictors and continuous model classes

### Assumptions

- A fixed measurable ambient class Theta of controlled finite-state hidden models is specified before evaluation. It may be continuous, with varying finite state dimensions, subject to measurable confidence sets/suprema. The actual controlled observation law has a member theta_star. Its initial prior is included, either directly or as a nuisance parameter.
- Actions depend on observable history and independent external randomness.
- A finite menu of global memory lengths, finite observation/action alphabets, and the known reward r(o,a) in [0,1] are fixed.
- At each step, q_t is a normalized next-observation probability vector chosen using only information available before that observation. Training, model selection, and hyperparameter selection used to produce q_t must all be predictable. The logged historical q_t values are immutable; one may not rescore the past with the final checkpoint.

Let Q_t be the product of these prequential predictions, including a normalized initial-observation prediction, and L_t(theta) the candidate causal observation likelihood. Define

C_t = {theta in Theta : L_t(theta) >= alpha Q_t}.

Then

P(theta_star in C_t for every t) >= 1-alpha.

**Proof.** Under theta_star, on its positive-probability histories, the conditional expected multiplier in Q_t/L_t(theta_star) is the sum of q_t(o) over the support of the true next-observation law, at most 1. Thus the ratio is a nonnegative supermartingale starting at 1. Ville's inequality controls its entire running maximum. Only the fixed true parameter is tested for this coverage assertion; no union over Theta is needed. If Q becomes zero, coverage is still valid but the confidence constraint becomes uninformative. Smoothing the predictor avoids that degeneracy.

There is no free statistical-complexity benefit: confidence-set width and certification power still depend on the predictor's accumulated log-loss regret and on identifiability, while optimization can be intractable.

This is a statement about inference over the full ambient class. A post-fit shortlist may be used as a computational proposal, but cannot replace C_t without a separate guarantee that the discarded region is infeasible. A parameter grid is an inner approximation and can invalidate the claim. A certified outer approximation and a certified upper bound on its envelope supremum are sufficient.

### Unknown initial prior

For fixed kernels theta and arbitrary initial prior b,

L_t(theta,b)=ell_t(theta)^T b,

where ell_t(theta)_j is the trajectory likelihood starting in hidden state j. Therefore

sup_b L_t(theta,b)=max_j ell_t(theta)_j.

Project the joint confidence set to kernels:

C_t^ker = {theta : max_j ell_t(theta)_j >= alpha Q_t}.

The actual kernels remain simultaneously covered with probability at least 1-alpha, without paying a cardinality penalty for the prior. This follows by retaining the actual (theta_star,b_star) before projection.

Caution: the profiled ratio Q_t/max_b L_t(theta,b) need not itself be a supermartingale. It is pointwise dominated, under each true prior, by that prior's valid likelihood-ratio martingale/supermartingale. For two permanent modes with Bernoulli(.1) and Bernoulli(.9) emissions, iid fair-coin Q, and the .1 mode as truth, the profiled ratio after O1=1 is 5/9, but its conditional expected next value is 205/81, approximately 2.531. Coverage follows by domination, not by claiming one-step supermartingale behavior for the profile.

### General test-process version

The same downstream result holds for C_t={theta:E_t(theta)<=1/alpha}, where for every theta, E_t(theta) is a valid nonnegative test supermartingale initialized at at most 1, or a genuine anytime-valid e-process, under that candidate and the allowed policy. Individually valid fixed-time e-values without an anytime argument do not suffice. This form is used by the erasure-model example.

### Window envelope and statistical composition

Use column-state conventions: T_a[s′,s]=P(s′|s,a), H[o,s]=P(o|s), and D_o=diag(H[o,:]). Let d=|O|, let J be the total number of predeclared (memory length, observation/action window, next action) contexts, and let gamma∈(0,1). Total variation means half the L1 distance.

For a window w=(o1,a1,...,a_(m-1),om), write

B_theta(w)=D_om T_(a_(m-1)) ... D_o2 T_a1 D_o1,
A_theta(w,a)=H T_a B_theta(w), d_j=1^T B_theta(w)e_j.

For d_j>0 set v_j=A_theta(w,a)e_j/d_j. All next-observation laws obtained from arbitrary start-of-window priors are exactly conv{v_j}: normalized prior weights are d_j b_j / sum_i d_i b_i; conversely every convex weight is attained by b_j proportional to weight_j/d_j. Consequently the exact diameter is max_(i,j) TV(v_i,v_j).

Set eta_t(w,a)=sup_(theta in C_t) max_(i,j) TV(v_i,v_j), using an empty-conditional-set convention only for impossible windows. Use a supremum unless attainment is established. A projection to kernels may be used, at the cost of its additional conservatism.

On the simultaneous truth-retention event, every sampled true predictive law and every other supported true history ending in w lies in the same true-model convex hull. Therefore their average differs from any such history by at most eta_t(w,a). Intersecting with the count event gives, simultaneously over times, contexts, and supported true histories,

TV(qhat_t(.|w,a), P_star(.|h,a)) <= min{1, b(N_t(w,a))+eta_t(w,a)}.

No cross-model diameter and no independence between events is required. Total failure probability is at most alpha+delta_data.

For a data-selected global m, define rho_t(m) as the maximum displayed radius on a verified state set covering all possible true windows. Freeze the empirical shift-state model. A uniformly epsilon_plan-optimal policy in that model satisfies, for every supported full history of length at least m,

V_star^*(h)-V_star^pihat(h)
<= min{1/(1-gamma), 2 gamma rho_t(m)/(1-gamma)^2 + epsilon_plan}.

**Planning proof.** Lift empirical value functions through the window map. They have span at most 1/(1-gamma). The one-step discrepancy of either the optimal Bellman operator or a fixed-policy operator is at most gamma rho/(1-gamma). Discounted contraction divides by 1-gamma. Comparing true and empirical optimal values and true and empirical policy values yields the two errors plus epsilon_plan.

## 3. Two different obstacles to nonvacuous certification

### 3.1 Unreachable-state augmentation is an envelope artifact

Take any controlled finite-state model and a finite menu of contexts. Add disconnected hidden chains, with zero initial mass, for each context (w,a) and each of two distinct final observation symbols. Each chain deterministically reproduces w under its prescribed actions and, after a, emits its assigned final symbol. Keep the original component's initial mass 1.

The augmented model has exactly the original observation law under every policy and at every horizon. But the two start-of-chain columns for each (w,a) have normalized next-observation distributions at TV distance 1. Hence its arbitrary-prior eta(w,a)=1 for every candidate context. This is a finite construction. A fixed family containing the original and augmented model has equal likelihoods forever and therefore keeps the unusable envelope forever.

This is **not** a claim that the actual observable process has long memory. It shows that the certificate's arbitrary-prior surrogate is not representation-invariant.

A safe refinement for a candidate with a fixed known initial prior is to restrict columns to states graph-reachable from that prior's support under some action sequence at an admissible window-start time. The true start-of-window posterior cannot leave this set; the convex-hull upper bound remains valid, though it need not remain an exact description of reachable priors. This removes disconnected zero-mass ghosts. It does not remove rare reachable states, and taking all priors after nuisance projection usually loses this refinement.

### 3.2 Rare reachable permanent modes are a statistical obstacle

Let P0 be iid Bernoulli(beta), and P1 be the equal mixture of P0 and iid Bernoulli(1-beta), 0<beta<1/2, with the component permanently selected at initialization. The same policy can be used in both models. Every fixed observation string recurs indefinitely within either component. P0 has zero predictive closure diameter; P1 has all-histories finite-window diameter 1-2 beta for every finite m.

For any event A that a procedure eventually certifies a closure below 1-2 beta,

P1(A) >= (1/2) P0(A).

If it certifies under P0 with probability at least 1-delta, it falsely certifies under P1 with probability at least (1-delta)/2. For delta<1/3, it cannot have both such power and delta-level soundness on this class. The argument concerns the eventual event and therefore cannot be escaped by waiting for more recurring local windows. This is established mixture indistinguishability, not a newly discovered impossibility principle.

Reachability pruning does not fix this example. The two modes have positive prior mass. Additional intervention/reset structure that genuinely distinguishes modes, a restriction excluding this mixture, or a weaker deployment-conditioned target is necessary.

## 4. Bounded learned state-changing example

This instantiates the repaired theorem rather than proposing a stand-alone toy-paper result.

### Model and what the learner sees

Hidden state X_t is binary; actions are binary and genuinely change latent dynamics:

X_(t+1)=X_t xor A_t xor Z_t, Z_t iid Bernoulli(theta), theta in [0,1/2].

The sensor reveals X_t with probability 0.6 and otherwise returns '?', independently. The erasure probability is known; the continuous transition parameter and initial hidden prior are unknown. The fitting and certification routines receive only observations and executed actions. The separate audit diagnostic evaluates the e-process at the simulator's true theta; that diagnostic is not used for inference or memory selection. There are no reset calls. This particular model is mixing for interior theta; the general soundness theorem does not assume mixing, and this example does not demonstrate success on irreversible dynamics.

When both adjacent observations reveal bits, the innovation z is observable as O_t xor A_t xor O_(t+1). With n previous informative transitions and k previous flips, the KT predictor uses q=(k+1/2)/(n+1) **before** the next observation.

A transition starting from '?' contributes test factor 1. Starting from a revealed bit, a next erasure contributes 1; a next revealed bit contributes q(z)/Bern_theta(z). Conditional expectation under the candidate is .4+.6 sum_z q(z)=1, or at most 1 at support boundaries. Thus the outcome-dependent availability of a revealed endpoint is correctly accounted for; no false predictability claim is made about the two-reveal indicator.

After n informative transitions, the e-process is

E_n(theta)= [B(k+1/2,n-k+1/2)/B(1/2,1/2)] / [theta^k (1-theta)^(n-k)].

Its inversion over the entire interval [0,1/2] gives a confidence interval by scalar concave-likelihood root finding. No sampled parameter grid is used.

Any possible window containing a revealed bit has a filtering matrix of rank at most one, so its predictive diameter is zero. Impossible windows are handled separately. An all-erasure window has exact diameter

eta_theta(w,a)=0.6 (1-2 theta)^m.

The supremum over the confidence interval is therefore attained at its lower endpoint, exactly. This is why the example is genuinely computable rather than hiding a nonconvex global optimization oracle.

### Frozen experiment and result

- Five fixed seeds: 664921, 1729, 20261003, 42, 99173
- 200,000 transitions per seed; no seed search, early stopping, or expensive training
- True theta=.32; simulator initial probability .83, not supplied to inference
- Uniform random actions in these long runs; a separate exact check uses adaptive actions
- alpha_model=delta_data=.025; total theorem failure level .05
- Lengths {1,2,3}, J=258, d=3; gamma=.5; target loss certificate .55
- Known reward r(o,a)=.95 1{o=1}+.05(1-a)

At 0, 2,000, and 20,000 transitions, all five runs abstained. At 200,000, all selected memory 2. Its bound ranged from 0.50805665 to 0.51698463. The confidence-interval lower endpoints ranged from .31059934 to .31335950; all intervals contained .32. Keeping the entire original theta range [0,.5] instead would yield the trivial capped bound 2 for every length, so parameter learning was necessary for this certificate.

Memory 1 bounds were approximately .977–.990. Memory 3 bounds were approximately .831–.875 because sparse contexts worsened the count term. The result is a genuine bias/coverage tradeoff rather than monotonically preferring the longest window.

The empirical memory-2 MDP has 18 states. Value iteration converged in 40 iterations for each seed, with Bellman residual below 7.0e-13 and a conservative numerical empirical-policy gap bound below 2.8e-12. Policies used both actions (12 states chose action 0; six chose action 1). No true-policy return comparison or learned-control advantage was measured.

### Independent checks and numerical caveat

`verify.py` additionally enumerates all positive-probability observation strings through eight observations using exact rational arithmetic under an observation-adaptive policy. At theta in {0,1/8,8/25,1/2}, the exact probability of ever excluding truth at nominal .05 is respectively 0, .01192085, .00278218, and .00026244. These finite checks corroborate, rather than prove, the anytime theorem.

Across 30,656 possible matrix/window cases, including exact zero support, the analytic envelope matched explicit normalized columns within 1.39e-16. Another 6,664 impossible candidate contexts were explicitly skipped rather than normalized.

The long-run outputs use double precision and outward root brackets with an extra 1e-10 margin. An independent 100-decimal-digit calculation verified all 15 nonempty reported interval enclosures, with positive outward log-threshold margins. These remain reproducible numerical illustrations of a rigorously defined exact-arithmetic certificate, not a formally verified floating-point implementation. No positivity tolerance is used to discard small columns.

## 5. Precise prior-art and originality assessment

- [Ortner, Maillard and Ryabko, 2014, OAMS](https://arxiv.org/pdf/1405.2652), equations (5)–(6), Lemma 1, and its setting: empirical transition confidence plus representation approximation error is already established. Their regret framework uses a finite weakly communicating underlying Markov representation and an average-reward objective. It is not identical to arbitrary POMDP histories with a discounted conditional certificate; nonetheless, adding a count term to an approximation term is not novel.
- [Subramanian, Sinha, Seraj and Mahajan, 2022, AIS](https://jmlr.org/papers/volume23/20-1165/20-1165.pdf), Definition 26 and Theorem 27: supplies the approximate-information-state/Bellman framework for discounted policy comparison. One must align their integral-probability-metric normalization with half-L1 TV. The planning step here is an instance, not a new principle.
- [Zhan, Uehara, Sun and Lee, 2022, CRANE/PSR](https://arxiv.org/pdf/2207.05738), equation (6) and its sample-complexity theorems: realizable partially observed model classes are estimated by likelihood confidence sets, including function approximation with class complexity. Their polynomial-learning result uses structural PSR assumptions, bracketing complexity and designed episodic exploration. It should not be paraphrased as a guarantee for an arbitrary learned finite ensemble on one continuing irreversible trajectory.
- [Emmenegger, Mutny and Krause, 2023, likelihood-ratio confidence sets](https://arxiv.org/html/2311.04402), Section 2 and Theorem 1: predictable estimator sequences generate anytime likelihood confidence sets. This directly precedes the learned-numerator repair. [Wasserman, Ramdas and Balakrishnan, 2020, Universal Inference](https://arxiv.org/abs/1912.11436) supplies the earlier universal-inference lineage. The no-cardinality-union coverage argument is established.

The normalized-column convex hull is an elementary linear-fractional image identity. Combining it with sequential model confidence and fixed-window visit confidence produces a transparent certificate, but no substantial general originality is established by this audit. The new work here is a careful composition, concrete failure modes, a computable continuous-parameter instantiation, and a sharper diagnosis of what a meaningful next result must overcome.

## 6. Remaining research target and stopping judgment

The next important question is not whether another scalar model can pass this certificate. It is whether one can produce a **representation-invariant, deployment-relevant closure certificate** for a genuinely learned controlled latent model, with:

1. An honest ambient model/representation class or a separately justified approximation allowance
2. A computable outer confidence set and certified envelope optimization
3. A reachable-history or deployment-probability target that avoids disconnected-state artifacts and states precisely which rare alternatives may be ignored
4. Intervention/coverage assumptions strong enough to identify the target, with a negative test on irreversible alternatives
5. A full-history filter and properly matched learned-model baselines

An unrestricted all-histories, single-trajectory theorem with high power cannot satisfy these requirements on the mixture class above. No such improvement is claimed here.

Reproduction: run `python verify.py`. The local execution record states that the protocol was fixed before the runs; the included hashes establish file integrity, not independent proof of chronology or public preregistration. All five runs and all checks are retained in `results.json` and `run.log`. A separate automated full replay reproduced the results file byte-for-byte; `high_precision.json` records the separate interval-endpoint check. The original prior artifacts were left unchanged.


## Reproduction and publication scope

From this directory, run `python verify.py` and then `python check_high_precision.py`. The scripts use Python 3, NumPy, SciPy, and mpmath; no external data, account, network connection, or accelerator is required. See [README.md](README.md) for the file map and checksums. The nominal 95% guarantee is per trajectory under the stated assumptions, not a simultaneous 95% claim across all five trajectories. The long runs report conservative numerical illustrations rather than a formally verified floating-point implementation.

This public report has no claimed journal acceptance, arXiv identifier, or DOI. No new open-source or Creative Commons license is granted by this publication; a component-specific license decision remains pending, without changing any existing repository or dependency terms.
