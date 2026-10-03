# Corrected finite-family baseline

Author: 潘奕成 (Yicheng Pan). Publication version 1.0, 2026-10-03. This companion states the fixed-family case with the causal-likelihood, exact-support, and supported-history clarifications used in [REPORT.md](REPORT.md). It is a composition of established sequential-inference and approximate-information-state arguments, not a claim of an unrestricted learned-ensemble guarantee or of new general principles. See the report for references, AI-assistance disclosure, limitations, and the continuous-class extension.

## Precise question

Can a learner certify a data-selected finite memory length from observable data without knowing the true latent states or the true member of a finite POMDP family, and without assuming latent mixing or reset access?

Answer: yes, conditionally on a fixed finite realizable model family, through the certificate below. The certificate may remain vacuous forever. It does not prove that irreversible contact dynamics can generally be learned from one trajectory.

## Setup

Finite actions A and observations O, with d=|O|. Observations arrive before actions. A full history is h_t=(o_1,a_1,...,a_{t-1},o_t). Reward r(o_t,a_t) is known and lies in [0,1]; a supplied planning objective is allowed, but hidden-state labels are not. Discount is gamma in (0,1).

Before seeing data, fix a finite family M of controlled finite-state POMDPs, including each candidate's transition kernels, emission kernels and initial distribution. Assume the true data-generating POMDP M* belongs to this family. The identity of M* and all its realized latent states are unknown. Latent transitions may have zeros, absorbing modes, and irreversible action effects. Candidate-state dimensions may differ.

Data are one continuing trajectory under any observable-history-based adaptive action rule. No reset, stationarity of the observed process, independent transition samples, or latent mixing is assumed. Fix a finite candidate set of global window lengths. For each length m, W_t^m contains exactly the last m observations and m−1 intervening actions. Use it only at t>=m. Its update is the ordinary recursive shift. Let J be the total number of (m,w,a) contexts over this fixed family.

A retrospectively trained arbitrary encoder or neural ensemble is not covered merely by calling it a candidate family. Realizability is an explicit substantive assumption; these checks cannot establish it.

## 1. Anytime model confidence

Let ell_t(M) be the causal observation log likelihood through time t, multiplying candidate next-observation probabilities along the executed action sequence and integrating out latent states. This is not the ordinary conditional likelihood given the entire adaptive action vector. Define

C_t={M: ell_t(M)>=max_M' ell_t(M')−log(|M|/delta_model)}.

With probability at least 1−delta_model, M* belongs to C_t simultaneously for all t.

Proof: for any fixed false M, the observable likelihood ratio L_t(M)/L_t(M*) is a nonnegative martingale, or supermartingale when supports differ, under M*. The adaptive action probabilities cancel because the action rule is the same measurable function of the observed past in every candidate. Ville's inequality gives crossing probability at most delta_model/|M|. A union bound proves the statement. No convergence or true-model identification is asserted.

## 2. Exactly computable predictive envelope

Use column-state conventions. For candidate M, let T_a[s',s]=P_M(s'|s,a), H[o,s]=P_M(o|s), and D_o=diag(H[o,:]). For

w=(o_1,a_1,...,a_{m-1},o_m),

write B_M(w)=D_{o_m}T_{a_{m-1}}...D_{o_2}T_{a_1}D_{o_1}.

For a next action a, define A_M(w,a)=H T_a B_M(w). Its jth column has mass d_j=1^T B_M(w)e_j. Discard exactly zero-mass columns. Normalize each remaining column:

v_j=A_M(w,a)e_j/d_j.

The set of next-observation laws obtainable by placing any prior b on the hidden state at the start of the window is exactly conv{v_j}. Indeed,

A_M b/(d^T b)=sum_j [d_j b_j/(d^T b)]v_j.

Conversely, every mixture lambda is attained by b_j proportional to lambda_j/d_j. Therefore its TV diameter is exactly

eta_M(w,a)=max_i,j TV(v_i,v_j).

Compute the data-dependent closure envelope

eta_t(w,a)=max_{M in C_t} eta_M(w,a).

Only the maximum INTRA-model diameter is needed. A cross-model diameter would unnecessarily charge uncertainty in the local conditional mean twice. On the likelihood-confidence event, eta_t upper-bounds the difference between the true next-observation laws at any two full histories with suffix w. The arbitrary-prior envelope can be much larger than the diameter of actually reachable histories, so this is conservative.

If no candidate or no relevant positive-mass columns remain, do not certify that context. Discard exactly zero-mass columns only; an arbitrary positive numerical cutoff is not a valid support rule. All claims concern true-supported causal histories, and impossible conditional laws are undefined.

## 3. Visit-count confidence under non-Markov data

For a fixed (m,w,a), let I_t indicate that W_t^m=w and A_t=a. This indicator is measurable before the next observation. Let p_t be the full-history conditional law of O_{t+1}. At N=n visits, let qhat_n be the empirical next-observation histogram and qbar_n the average of these n conditional laws.

Define

b(n)=min{1, sqrt(log(pi^2 J 2^d n^2/(6 delta_data))/(2n))}, n>=1,

and b(0)=1. With probability at least 1−delta_data, simultaneously over all contexts and times,

TV(qhat_n,qbar_n)<=b(n).

Proof: for every subset E of observations and every real lambda,

exp(lambda sum I_t[1{O_{t+1} in E}−p_t(E)]−lambda^2 N/8)

is a nonnegative supermartingale by conditional Hoeffding. Stop at the nth visit, optimize lambda and union-bound over n, J contexts and 2^d subsets. The sum of 6/(pi^2 n^2) is one. The maximum positive subset discrepancy is TV, with complements covering the opposite sign. No latent Markov or mixing property is used.

Combining the two probability events, for every time t, candidate m and true-supported full history h of length at least m with suffix w,

TV(qhat_t(.|w,a),P_M*(O_next in .|h,a)) <= rho_t(w,a),

rho_t(w,a)=min{1,b(N_t(w,a))+eta_t(w,a)}.

This remains valid when t is a stopping time and the global m is selected from the fixed candidate family after observing the data. qbar_n is a random average of conditional laws, not a fixed population parameter. Counts alone do not certify the closure error eta.

## 4. Planning consequence

Freeze the empirical model at time t. For a chosen global m, update its state by shifting w and appending (a,o), and use qhat_t for next-observation probabilities. Unvisited contexts get an arbitrary stochastic prediction and radius one. Let rho_t(m)=max_w,a rho_t(w,a) on a verified support-closed state set covering every possible true window. Unvisited but possible contexts have radius one. If support closure or nonempty relevant conditional sets cannot be verified, abstain.

If pihat is epsilon_plan-optimal uniformly in the empirical discounted MDP, then on the same event, for every actual full history h of length at least m,

V_M*^*(h)−V_M*^pihat(h) <= 2 gamma rho_t(m)/(1−gamma)^2 + epsilon_plan.

The comparator may use the entire history. Proof: lift the empirical value functions to full histories through the window map. The one-step Bellman discrepancy is at most gamma rho/(1−gamma), since values lie in [0,1/(1−gamma)]. Discounted contraction bounds true-versus-empirical optimal and fixed-policy values separately, giving the two factors. This is standard simulation/AIS reasoning, not a new principle. A reward error epsilon_r adds 2 epsilon_r/(1−gamma). Rewards in [−1,1] require twice the transition-error term.

A practical certificate can select the smallest global m whose displayed bound meets a specified tolerance, or abstain if none does. This statement concerns soundness, not the probability or speed of certification. It does not claim a neural or planning-performance improvement over a correctly specified full-history filter.

