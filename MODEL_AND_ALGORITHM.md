# Theory: hypergraph influence maximization with GIP

## 1. Problem setting

Let

\[
\mathcal H=(V,E)
\]

be a finite weighted hypergraph. Nodes \(v\in V\) are municipalities and
hyperedges \(e\in E\) are participated companies. Unlike an ordinary graph, a
single company can connect any number of municipalities simultaneously.

Let \(n=|V|\), \(m=|E|\), and define the nonnegative weighted incidence matrix
\(B\in\mathbb R_+^{n\times m}\) by

\[
B_{ve}=
\begin{cases}
q_{ve}, & v\in e,\\
0, & v\notin e,
\end{cases}
\]

where \(q_{ve}\) is the strength of the municipality--company relation, here
the municipality's ownership quota.

Two additional parameter vectors characterize the model:

\[
\kappa\in\mathbb R_+^m,
\qquad
\omega\in\mathbb R_+^n.
\]

The hyperedge parameter \(\kappa_e\) controls how strongly company \(e\)
transmits influence. The node parameter \(\omega_v\) controls how much
activation of municipality \(v\) is worth in the objective. In the present
interpretation, company size determines \(\kappa_e\), while municipality
population determines \(\omega_v\).

This separation is substantive:

- \(B\) determines which municipality--company channels exist and their
  strength;
- \(\kappa\) affects the propagation dynamics;
- \(\omega\) affects only the valuation of a propagation path.

## 2. Generalized influence propagation

Let \(x^{(t)}\in\mathbb R_+^n\) denote municipality activation intensity at
discrete time \(t\). The state is continuous: \(x_v^{(t)}>0\) means that
municipality \(v\) is active, and the magnitude records its intensity.

For a seed set \(S\subseteq V\), the initial state is

\[
x_v^{(0)}=a_0\mathbf 1\{v\in S\},
\]

where \(a_0>0\) is the common seed intensity.

### 2.1 Municipality-to-company aggregation

At time \(t\), every company aggregates the activation of all its municipal
owners:

\[
p^{(t)}=B^\top x^{(t)},
\qquad
p_e^{(t)}=\sum_{u\in V}B_{ue}x_u^{(t)}.
\]

The scalar \(p_e^{(t)}\) is the pressure received by hyperedge \(e\).

### 2.2 Hyperedge activation and transmission

Let \(\tau\ge0\) be the hyperedge stress threshold. Hyperedge \(e\) transmits
only when its aggregate pressure strictly exceeds \(\tau\):

\[
g_e^{(t)}
=\mathbf 1\{p_e^{(t)}>\tau\}.
\]

Its transmitted pressure is

\[
r_e^{(t)}
=\kappa_e p_e^{(t)}g_e^{(t)}.
\]

The activation gate depends on aggregate pressure, not directly on
\(\kappa_e\). Thus several municipalities may jointly activate a company even
when none of them could activate it alone.

### 2.3 Company-to-municipality redistribution

The shared gate uses pressure from **all** shareholders, including the recipient.
Only the returned contribution excludes the recipient's own old state. Set
\(a^{(t)}=\kappa\odot g^{(t)}\) and let \(B^{\odot2}\) be the elementwise
square of the incidence matrix. Then

\[
d^{(t)}=B^{\odot2}a^{(t)},\qquad
z^{(t+1)}=B(a^{(t)}\odot p^{(t)})-x^{(t)}\odot d^{(t)}.
\]

The independent componentwise definition is

\[
z_v^{(t+1)}
=\sum_{e\in E}B_{ve}\kappa_e g_e^{(t)}
  \sum_{u\ne v}B_{ue}x_u^{(t)}.
\]

All terms use the same old state. There are no recipient-specific gates, no
incidence deletion, and no diagonal modification of \(B\). For example,
\(B=(1,1)^\top\), \(x=(1,1)\), \(\kappa=1\), \(\tau=1.5\) gives
\(z=(1,1)\), even though each leave-one-out pressure is below the gate.
The gate is strict: pressure equal to \(\tau\) does not activate a company.

Production evaluation uses sparse matrix-vector products and caches
`B.multiply(B)` in a private snapshot of an unchanged incidence matrix.
It never constructs a dense municipality-by-municipality adjacency matrix.
The raw result is mathematically nonnegative. Negative cancellation roundoff
is set to zero only within a per-node tolerance of
`8 * float64_eps * operation_count * max(abs(total), abs(own))`, plus eight
smallest subnormal units. The operation count includes row and column
reduction lengths. Material negatives and non-finite arithmetic raise errors.
Ownership units, company parameters, and municipal weights are not normalized
by the diffusion evaluator.

### 2.4 GIP threshold operator

The raw input \(z^{(t+1)}\) is transformed by a lower activation threshold and
an upper saturation threshold. Define

\[
T_{L,U}(z)=
\begin{cases}
0, & z<L,\\
z, & L\le z\le U,\\
U, & z>U.
\end{cases}
\]

The time-dependent thresholds at propagation step \(j\ge1\) are

\[
L_j=(\theta_L\alpha)^j l_0,
\qquad
U_j=\theta_H\theta_L^{j-1}\alpha^j h_0,
\]

with nonnegative \(l_0,h_0,\theta_L,\theta_H,\alpha\) and \(U_j\ge L_j\).
When the parameters are positive, the latter condition is equivalent to

\[
\theta_H h_0\ge\theta_L l_0.
\]

The next state is

\[
x_v^{(t+1)}
=T_{L_{t+1},U_{t+1}}\!\left(z_v^{(t+1)}\right).
\]

Equivalently, the update is \(x^{(t+1)}=T_{L_{t+1},U_{t+1}}(z^{(t+1)})\).
For interpretation only, the conditional municipal operator is

\[
W_{\mathcal H}^{(t)}
=B\operatorname{diag}(a^{(t)})B^\top
-\operatorname{diag}(B^{\odot2}a^{(t)}).
\]

Its diagonal is zero. If all pressured companies are active, replace
\(a^{(t)}\) with \(\kappa\). Binary incidence columns with exactly two
endpoints and \(\tau=0\) therefore reduce to ordinary GIP on the
zero-diagonal weighted graph adjacency, with matching bounds and scoring.

## 3. Consequences of the hypergraph operator

The operator has several important theoretical consequences.

### Collective activation

Because the stress gate is applied after aggregation, influence can exhibit
complementarity. Several municipalities can jointly push a company above
\(\tau\), even though each municipality's contribution is individually
insufficient.

### Broadcast within a hyperedge

Once a company activates, every incident municipality can receive influence
from its other shareholders in the same propagation step. A hyperedge therefore represents a
single many-to-many interaction, not a sequence of independent pairwise
events.

### Additive overlapping channels

If two municipalities share several companies, their contributions through
those companies add. Hyperedge overlap can therefore amplify propagation.

### No direct self-return

A company with one municipal owner returns zero influence. With two owners
and only one active, only the other municipality receives the first return.
A return through that other municipality on a later step is still permitted.
Removing direct self-return changes neither the shared gate nor the incidences.

### Dependence on incidence units

For \(u\ne v\), transmission through a shared company contains the product

\[
B_{ue}B_{ve}=q_{ue}q_{ve}.
\]

Hence changing quotas from percentages to fractions rescales the two-stage
operator quadratically. The incidence units and the threshold parameters must
therefore be defined jointly. In the current specification, quotas enter in
their source units and are not normalized inside the propagation operator.

### Non-progressive activation

The model recomputes \(x^{(t+1)}\) entirely from \(x^{(t)}\). Activation is not
permanent: a municipality may activate, deactivate, and potentially reactivate
later. This differs from progressive cascade models in which activation is
absorbing.

## 4. Propagation horizon and stopping

`gip(..., horizon=T)` performs exactly \(T\) updates and stores/scores \(T+1\)
states, including \(T=0\). Tolerance and consecutive-state equality are ignored.
Use the same fixed horizon for all candidate seed sets in theoretical comparisons.
The experiment runner and active influence notebook default to \(T=20\);
this is a declared finite horizon, not a claim that the omitted tail is negligible.

With `horizon=None`, `max_iter` is a finite positive \(J_{\max}\). Before step
\(j\ge1\), numerical early stopping tests

\[
\left\|(1-\gamma)^{j-1}x^{(j-1)}\right\|_2\le\varepsilon.
\]

Otherwise the evaluator computes and scores step \(j\), up to \(J_{\max}\)
updates. Equality of two successive states is never a stopping rule: bounds
may change later, and positive stationary states still contribute discounted
value. Results report `stopping_reason` (`fixed_horizon`, `tolerance`, or
`max_iter`) and `iterations` (actual updates).

The norm criterion is a numerical truncation, **not a proven bound on omitted
objective value**. Its stopping time can depend on the seed set and on gamma.
Fixed-horizon monotonicity must not automatically be claimed for these truncated
scores. Gamma never attenuates the propagation state itself.

## 5. Generalized influence-maximization objective

Let \(x^{(t)}(S)\) be the trajectory generated by seed set \(S\). For a fixed
horizon \(T\), define population-weighted, time-discounted cumulative spread with
\(0<\gamma<1\) and discount \(1-\gamma\),

\[
F_T(S)
=\sum_{t=0}^{T}(1-\gamma)^t\omega^\top x^{(t)}(S).
\]

With an endogenous stopping rule, the corresponding objective is

\[
F(S)
=\sum_{t\in\mathcal T(S)}(1-\gamma)^t\omega^\top x^{(t)}(S),
\]

where \(\mathcal T(S)\) is the stored trajectory of \(S\).

The generalized influence-maximization problem is

\[
\boxed{
\max_{S\subseteq V}F(S)
\quad\text{subject to}\quad
|S|=k.
}
\]

This is a fixed-cardinality intervention: exactly \(k\) municipalities receive
the initial seed intensity. No connectivity condition is imposed on \(S\).

The objective makes three deliberate choices:

1. **Population weighting.** Activation of municipality \(v\) is valued by
   \(\omega_v\), but \(\omega_v\) does not affect its ability to transmit.
2. **Cumulative valuation.** A municipality contributes at every time step at
   which it is active, rather than only the first time it activates.
3. **Intensity valuation.** The contribution is
   \(\omega_v x_v^{(t)}\), not merely a binary reached/not-reached indicator.

The initial state \(t=0\) is included. Consequently, node weights affect both
the direct value of choosing a seed and the value of its subsequent diffusion.
The score starts at `dot(omega, x0)` and adds the discounted value after each
update. Changing omega affects only scores; changing gamma also leaves any
fixed-horizon trajectory unchanged. Neither is applied to B, pressure, or gates.

Some propagation pseudocode in the graph paper accumulates only steps t >= 1;
the mathematical graph IM objective includes t=0. With unit weights, common
a0, and fixed budget b, omitting seeds subtracts the constant b*a0. With
heterogeneous omega it subtracts a seed-dependent amount and can change rankings.
All objectives and ablation baselines in this repository use t=0 exactly once.

## 6. Relation to classical influence maximization

Classical IM commonly maximizes the expected number of distinct nodes
activated by a stochastic progressive cascade. The present problem differs in
four ways:

- propagation is deterministic;
- interactions are mediated by hyperedges;
- activation is continuous and non-progressive;
- spread is a weighted, time-discounted cumulative intensity.

For a fixed horizon, the propagation map is coordinatewise nondecreasing when
\(B,\kappa,\omega\) are nonnegative and \(U_t\ge L_t\). Indeed:

- \(B^\top x\) is nondecreasing in \(x\);
- each shared gate \(\mathbf 1\{p_e>\tau\}\) is nondecreasing;
- each leave-one-out returned sum \(\sum_{u\ne v}B_{ue}x_u\) is nonnegative
  and nondecreasing;
- their product and the sum with nonnegative coefficients preserve order;
- \(T_{L,U}\) is nondecreasing.

It follows by induction that

\[
x^{(0)}\le \widetilde x^{(0)}
\quad\Longrightarrow\quad
x^{(t)}\le\widetilde x^{(t)}
\]

for every fixed \(t\), and therefore \(F_T\) is monotone under seed-set
inclusion.

Monotonicity does **not** imply submodularity. The stress gate directly creates
increasing returns. Consider a company for which one seed contributes pressure
\(q\) and

\[
q\le\tau<2q.
\]

One seed alone does not activate the company, while two seeds together do.
The marginal propagation gain of the second seed is therefore larger when the
first seed is already present. This violates the diminishing-returns property
required for submodularity.

Upper caps can create diminishing returns in other regions, but they do not
remove the threshold-induced complementarities globally. Hence the classical
greedy \(1-1/e\) approximation guarantee for monotone submodular IM does not
apply without additional restrictions.

The gates and activation floors also make the objective discontinuous and
nonconvex. The optimization should therefore be treated as a black-box
combinatorial problem.

## 7. Parameter interpretation

The theory requires only nonnegative \(\kappa_e\) and \(\omega_v\). A useful
empirical parameterization is

\[
\kappa_e
=\mu_E
\frac{b_e^{\beta_E}}
{|E|^{-1}\sum_{f\in E}b_f^{\beta_E}},
\qquad
\omega_v
=\mu_V
\frac{p_v^{\beta_V}}
{|V|^{-1}\sum_{u\in V}p_u^{\beta_V}},
\]

where \(b_e>0\) is a company-size score and \(p_v>0\) is population.

The exponents control heterogeneity:

- \(\beta_E=0\) makes all companies equally transmissive;
- \(\beta_E>0\) introduces company-size differences;
- \(\beta_V=0\) gives an unweighted node objective;
- \(\beta_V>0\) values municipalities according to relative population.

The constants \(\mu_E\) and \(\mu_V\) set the respective means. Multiplying all
node weights by the same positive constant rescales the objective without
changing its maximizer. Multiplying all hyperedge weights can change the
trajectory because propagation is nonlinear and compared with fixed thresholds.

The threshold-scale parameter \(\alpha\) is distinct from \(\kappa\):
\(\alpha\) controls the time-dependent lower and upper GIP thresholds, whereas
\(\kappa_e\) controls relative transmission through company \(e\).

## 8. NaDS search over fixed-budget seed sets

The diffusion objective is expensive and lacks a general submodular structure,
so the implementation uses NaDS as a black-box exchange search.

For a seed set \(S\) of size \(k\), define the \(r\)-exchange neighborhood

\[
\mathcal N_r(S)
=
\left\{
(S\setminus R)\cup A:
R\subseteq S,\;
A\subseteq V\setminus S,\;
|R|=|A|=r
\right\}.
\]

Every candidate in \(\mathcal N_r(S)\) has the same cardinality as \(S\).
Given exchange-distance parameter \(d\), the broad neighborhood is

\[
\mathcal N_{\le d}(S)
=
\bigcup_{r=1}^{\lfloor d/2\rfloor}\mathcal N_r(S).
\]

Thus \(d=2\) permits one-for-one exchanges, while \(d=4\) also permits
two-for-two exchanges.

Starting from \(S_0\), NaDS repeatedly:

1. searches a one-for-one exchange neighborhood;
2. accepts a candidate only if it strictly improves \(F\);
3. if no one-for-one improvement is found, searches the broader
   \(\mathcal N_{\le d}(S)\);
4. terminates when neither phase finds an improvement.

The implementation uses a sufficient-improvement parameter \(\xi_\ell\), where
\(\ell\) indexes search iterations. A
neighborhood scan may stop early once it finds

\[
F(S')>(1+\xi_\ell)F(S).
\]

If the accepted improvement is positive but no larger than this threshold,
\(\xi_\ell\) is reduced:

\[
\xi_{\ell+1}=\delta\xi_\ell,
\qquad
0<\delta\le1.
\]

A finite memory buffer avoids repeated objective evaluations. Time limits,
iteration limits, and per-phase neighbor limits are computational truncations,
not additional feasibility conditions on the mathematical IM problem.

### Algorithmic properties

The search has the following guarantees:

- **budget preservation:** every accepted set contains exactly \(k\) seeds;
- **strict ascent:** accepted objective values form a strictly increasing
  sequence;
- **finite termination:** without computational truncation, strict ascent over
  the finite family of \(k\)-subsets must terminate;
- **local optimality:** if the relevant neighborhoods are exhaustively searched
  at termination, the returned set is locally optimal with respect to those
  exchanges.

It does not provide a general global-optimality or approximation guarantee.
Initialization, neighbor order, exchange radius, and computational budget may
therefore affect the returned local optimum.

The current theoretical formulation imposes neither a connectivity constraint
on the seed set nor an MG phase. If such restrictions are desired, they define
a different feasible set or a different search heuristic and should be stated
separately.

## 9. Core modeling assumptions

The model rests on the following assumptions:

1. **Static hypergraph during diffusion.** Incidences, quotas, and parameters do
   not change over a propagation trajectory.
2. **Synchronous deterministic updates.** All companies aggregate the same
   current node state, and all municipalities receive the resulting company
   outputs simultaneously.
3. **Nonnegative influence.** The model contains amplification, thresholding,
   and saturation, but no negative or inhibitory influence.
4. **Symmetric incidence mechanism.** The same \(B_{ve}\) is used on the
   municipality-to-company and company-to-municipality legs.
5. **Collective hyperedge gate.** A company activates according to aggregate
   pressure from all incident municipalities.
6. **Global GIP thresholds.** At a given time step, all municipalities share the
   same lower threshold and upper cap.
7. **Non-progressive state.** Activation is recomputed rather than permanently
   retained.
8. **Population as welfare weight.** Population changes valuation but not
   propagation.
9. **Discounted cumulative intensity as spread.** Repeated activation and
   activation magnitude are counted with time discount (1-gamma)^t, including seeds.
10. **Homogeneous seed treatment.** Every selected municipality receives the
    same initial intensity unless the intervention model is explicitly
    extended.
11. **Unrestricted fixed budget.** Any \(k\)-node set is feasible unless further
    cost, geographic, or policy constraints are added.
12. **Associational interpretation.** The ownership network defines potential
    channels of influence; the model alone does not establish causal effects.

Under these assumptions, the object being optimized is best described as
**population-weighted, time-discounted cumulative generalized influence on a
weighted hypergraph without direct self-return**.
