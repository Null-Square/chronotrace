# Counterfactual Training-History Observability

Status: theory/methodology note. This is not a paper claim and does not use held-out confirmation codebooks.

## 1. Finite forensic model

Let the finite candidate history set be

`W = {w_1, ..., w_m}`.

Let `q_1, ..., q_p` be allowed forensic probe coordinates. A probe coordinate may be a passive observation, an infinitesimal susceptibility coordinate, or a deterministic continuation followed by an observation. Define

`H[i,j] = R_{q_j}(w_i)`.

The row `H[i,:]` is the forensic response fingerprint of history `w_i` under the chosen access regime.

This representation deliberately separates two questions:

1. **information:** are two histories distinguishable under the allowed probe family at all?
2. **algorithm:** how many probe coordinates are needed to preserve those distinctions?

It does not assume that the complete trainer state is observable from weights, and it does not claim that low response rank holds for language models.

## 2. Exact indistinguishability

Define

`w_i ~ w_k  iff  H[i,:] = H[k,:]`.

This is an equivalence relation. Each equivalence class is an impossibility class for every decoder restricted to the supplied probe family: identical response rows provide identical evidence.

### Markov-state boundary

Suppose every probe is a deterministic map of the same complete Markov state `s`, followed by the same observation rule. If two histories produce exactly the same complete state, then all future deterministic responses are identical.

Therefore active probing cannot recover history that has been erased from the complete state.

For reset-SGD experiments where the future operator depends only on current weights, exact equality of weights is such an impossibility case. If optimizer moments or other hidden trainer variables are retained, equal weights need not imply equal complete states and future learning responses may differ.

This distinction is essential: the active-observability hypothesis is that history can be hidden from a **coarse current observation** while remaining encoded in the state, not that deterministic challenges can resurrect information absent from the state.

## 3. Finite probe-basis theorem

### Proposition 1 — rank-sized distinguishing basis

Let `H` be the `m x p` response matrix with rank `r`. Then there exists a set `J` of exactly `r` columns such that

`H[i,J] = H[k,J]  iff  H[i,:] = H[k,:]`

for every pair of candidate histories `i,k`.

#### Proof

Choose `r` columns of `H` forming a basis of the column space; call the resulting matrix `H_J`. Every omitted column `H[:,j]` is a linear combination of the selected columns:

`H[:,j] = H_J a_j`.

If rows `i` and `k` agree on the selected columns, then

`H_J[i,:] = H_J[k,:]`.

Multiplying both rows by any coefficient vector `a_j` gives equal entries in every omitted column. Hence the full rows agree. The reverse implication is immediate. QED.

### Consequence

If all `m` candidate rows of the full response matrix are distinct, at most `r` linearly independent response coordinates are sufficient to distinguish all candidates.

This is an existence theorem, not a claim that a given physical probe set is cheap to construct or that the basis is statistically well-conditioned.

## 4. Noise certificate

### Proposition 2 — nearest-response robustness

Let selected candidate response rows be `z_1, ..., z_m` and define

`delta = min_{i != k} ||z_i - z_k||_2`.

If the true history is `i` and the measured response is

`y = z_i + e`

with

`||e||_2 < delta / 2`,

then Euclidean nearest-row decoding uniquely returns history `i`.

#### Proof

The true-row distance is `||y-z_i|| = ||e|| < delta/2`.

For every competitor `k`, the reverse triangle inequality gives

`||y-z_k|| >= ||z_i-z_k|| - ||e|| >= delta - ||e|| > delta/2`.

Therefore every competitor is farther than the true row. QED.

A zero `delta` is an exact non-identifiability certificate for at least one history pair.

## 5. Minimal active toy falsifier

Use state `(x,y)` and base `(0,0)` with deterministic stages

- `A(x,y) = (x+1, y)`
- `B(x,y) = (x, y+x)`

Then

- `AB -> (1,1)`
- `BA -> (1,0)`.

Under the passive observation

`h(x,y)=x`,

both histories are identical: `h(AB)=h(BA)=1`.

Now apply challenge

`C(x,y)=(x+y,y)`.

Then

- `h(C(AB))=2`
- `h(C(BA))=1`.

Thus the histories are passively indistinguishable under `h` but actively distinguishable under the same observation after a controlled continuation.

This example proves possibility only. It does not show that real training histories exhibit such a separation.

## 6. First-order learning-response coordinates

For a finished parameter state `theta`, stage losses `L_A, L_B, ...`, and a small reset-SGD challenge on stage `B`,

`theta' = theta - epsilon * g_B(theta)`,

where `g_B = grad L_B`, Taylor expansion gives

`L_A(theta') - L_A(theta) = -epsilon <g_A(theta), g_B(theta)> + O(epsilon^2)`.

Define the first-order susceptibility matrix

`S_AB(theta) = <g_A(theta), g_B(theta)>`.

The passive loss vector asks what the endpoint currently knows. `S` asks how those losses would change under infinitesimal future learning.

This is the cheapest nontrivial active-response object to test on the already-observed Pythia pilot because it requires gradient evaluation but no additional chronology training probes.

## 7. Falsification ladder before held-out confirmation

Do not expose the four frozen confirmation codebooks during methodology development.

1. **Algebra gate:** executable tests must verify Propositions 1 and 2, exact non-identifiability, and the passive/active toy separation.
2. **Recency gate:** on the already-spent Pythia pilot, active-response features must be compared against the exact-replay marginal-loss baseline, not only against K2.
3. **First-order gate:** measure the endpoint susceptibility matrix on all 24 already-spent pilot histories. Report response rank, minimum pair separation, and whether susceptibility adds distinguishability beyond the passive loss vector.
4. **Decoder gate:** only if susceptibility contains additional stable information, construct a chronology decoder using a frozen candidate-generation rule. Do not evaluate a decoder by giving it the target's own labeled response row as a reference.
5. **Finite-challenge gate:** if first-order susceptibility is promising, validate it against actual small continuation steps and verify the expected first-order error scaling.
6. **Held-out gate:** freeze the complete active-probe protocol and aggregate rule before any model output from confirmation seeds is observed.

## 8. What would falsify the breakthrough direction early

The direction should be downgraded if any of these occur on the spent pilot:

- susceptibility rows are effectively functions of the passive stage-loss vector and add no pairwise distinctions;
- the susceptibility response matrix is numerically high-rank with poor separation, offering no probe-compression advantage;
- distinctions disappear under exact finite continuation even when the infinitesimal calculation predicts them;
- a response-augmented decoder does not beat both K3 endpoint geometry and marginal recency without target-label leakage;
- the useful response coordinates require factorial candidate simulation, eliminating the proposed scaling advantage.

The goal is not to preserve the hypothesis. The goal is to identify as cheaply as possible whether active training-history observability is a real additional information channel.