# ChronoTrace Results and Evidence Ledger

This file is the reviewer-facing index of completed ChronoTrace evidence. It does not replace the per-experiment result reports; it summarizes them and states the claim boundary that follows from each one.

## Status vocabulary

- **Positive controlled result** — the predeclared controlled mechanism/reconstruction test succeeded in its stated setting.
- **Negative / confounded result** — the target claim was not isolated because a control or gate failed.
- **Design-only negative result** — a design-selection pilot failed its admissibility gate; it is not a confirmatory chronology test.
- **Pending** — not yet established and must not be implied by current results.

## Evidence ledger

| Result | Status | Key evidence | Claim supported | Main source |
| --- | --- | --- | --- | --- |
| Phase-0 v1 Pythia-70M AB/BA discovery | **Negative / confounded** | Forensic detector BA/AUROC `1.000/1.000`; capability-only baseline also `1.000/1.000`; all 8 matched seed pairs failed the predeclared capability-equivalence gate; maximum control-margin gap `13.953095368496685`. Confirmation seeds were not run. | The endpoint strongly reveals recent-stage/capability differences. It does **not** isolate a non-trivial historical path signature. | [`results/PHASE0_V1.md`](results/PHASE0_V1.md) |
| Phase-0b shuffled-union terminal washout | **Design-only negative** | No tested `C` duration (`50`, `150`, `300`) passed the `<= 1.0` capability gate for every matched seed. At `C=300`, Order-Witness BA was `0.500` while capability-only BA was `0.833`. Confirmation seeds remained untouched. | A simple balanced common tail is not an adequate design for isolating chronology under endpoint capability equivalence. | [`results/PHASE0B_WASHOUT_PILOT.md`](results/PHASE0B_WASHOUT_PILOT.md) |
| Smooth nonlinear inverse-commutator theorem gate | **Positive controlled result** | Remainder slope `2.99346` (`O(eta^3)` target), held-out AB/BA behavior-gap slope `2.06531` (`O(eta^2)` target), endpoint-displacement slope `0.95967` (`O(eta)` target), all 6 permutations recovered at every tested step size. | The algebraic second-order chronology construction behaves as predicted in a smooth nonlinear system. | [`results/commutator_macro_gate.md`](results/commutator_macro_gate.md) |
| Tiny causal-transformer commutator gate | **Positive controlled result** | `1,032` trainable parameters; remainder slope `3.00290`; held-out loss-gap slope `1.98968`; displacement slope `0.99879`; all 6 permutations recovered across the tested step-size sweep; smallest-step ChronoScores `+0.99909` / `-0.99513`. | The inverse second-order chronology geometry survives a genuine causal-transformer parameterization and language-model loss in a deterministic small model. | [`results/commutator_macro_gate.md`](results/commutator_macro_gate.md) |
| Multi-update finite macro-stage operator gate | **Positive controlled result** | Differential macro decoder recovered `6/6` histories for `1,2,4,8,16,32,64` updates/stage; the one-step micro HVP decoder first lost perfection at `2` updates/stage. | Treating complete training stages as finite maps extends reconstruction beyond the reliable one-step HVP regime. | [`results/commutator_macro_gate.md`](results/commutator_macro_gate.md) |
| Exact finite-pair interaction gate | **Positive controlled result** | Finite-pair decoder recovered `6/6` histories for every tested stage length through `256` updates/stage; micro HVP first failed at `2`, differential macro first failed at `128`. No HVP, double backward, or finite-difference epsilon is used. | Exact singleton + ordered-pair interactions are the strongest current controlled white-box chronology decoder and justify the next model-scale gate. | [`results/finite_pair_gate.md`](results/finite_pair_gate.md) |
| Sufficient nearest-signature certificate | **Positive only through 32 updates/stage** | `2 ||r_high|| / delta_min < 1` holds through `32` updates/stage; it is `1.4289`, `2.0025`, `2.0173` at `64`, `128`, `256`. | The certificate guarantees the early finite-pair sweep only. Recovery after `32` is empirical in this controlled system, not certified by this bound. | [`results/finite_pair_gate.md`](results/finite_pair_gate.md) |
| Realistic Pythia-scale finite-pair chronology reconstruction | **Pending** | Not yet run as the fixed next gate. | No large-model chronology claim follows from the current finite-pair result. | [`DECISIONS.md`](DECISIONS.md) |
| Seed-held-out capability-matched path persistence on Pythia | **Pending** | Phase-0 v1 failed the capability gate and Phase-0b did not produce an admissible washout design. | H1/H3 remain unresolved at realistic model scale. | [`HYPOTHESES.md`](HYPOTHESES.md) |
| Black-box chronology transfer / unknown training recipes | **Pending** | Outside the completed controlled gates. | Not established. | [`HYPOTHESES.md`](HYPOTHESES.md) |

## Strongest current reconstruction result

The finite-pair decoder models each candidate training stage as a finite map `F_i` from a known base checkpoint `theta_0`. For every ordered pair it measures

```text
I_{j<-i} = F_j(F_i(theta_0)) - theta_0 - Delta_i - Delta_j
```

where `Delta_i = F_i(theta_0) - theta_0`.

After caching the `N` singleton endpoints, the full ordered-pair table requires `N(N-1)` additional stage executions, for exactly `N^2` stage executions total. This replaces factorial replay of complete histories with a second-order interaction model.

### Fixed recovery sweep

| Updates/stage | Micro HVP | Differential macro | Finite pair | Max singleton displacement |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6/6 | 6/6 | 6/6 | `0.01085` |
| 2 | 4/6 | 6/6 | 6/6 | `0.02158` |
| 4 | 2/6 | 6/6 | 6/6 | `0.04271` |
| 8 | 2/6 | 6/6 | 6/6 | `0.08384` |
| 16 | 3/6 | 6/6 | 6/6 | `0.16262` |
| 32 | 5/6 | 6/6 | 6/6 | `0.31768` |
| 64 | 4/6 | 6/6 | 6/6 | `0.61171` |
| 128 | 2/6 | 4/6 | 6/6 | `1.15237` |
| 256 | 1/6 | 3/6 | 6/6 | `2.08180` |

The finite-pair decoder has no failure in the fixed sweep. This is an empirical controlled result on the deterministic `1,032`-parameter causal transformer with known stages, known base checkpoint, full weights, and plain SGD.

## Controlled mechanism evidence

The earlier commutator experiments establish why chronology information can exist at all in this setting.

For two small non-commuting updates, the order-dependent endpoint difference is predicted to appear at second order while the shared displacement remains first order. The smooth nonlinear experiment and tiny causal-transformer experiment both show the expected asymptotic scaling, and the transformer experiment recovers every A/B/C permutation in the tested local regime.

The finite macro-stage experiment then replaces local one-step gradients/HVPs with complete stage maps and finite-difference Jacobian interactions. It retains perfect six-order recovery through `64` updates/stage. The finite-pair experiment removes the derivative epsilon entirely and remains perfect through `256` updates/stage.

## Negative evidence that constrains the project

### Phase-0 v1

The first Pythia-70M endpoint experiment found perfect AB/BA classification, but the ordinary capability baseline was also perfect. Every matched seed pair violated the predeclared capability-matching gate. This is recorded as a clean confounded/negative result for the intended path-memory claim.

The confirmatory seeds `101`, `103`, `107`, and `109` were not trained, sealed, or inspected after the discovery gate failed.

### Phase-0b

A deterministic shuffled union of A and B examples was tested as a common terminal stage at `50`, `150`, and `300` updates using design-only seeds. No duration passed the capability gate. At `300`, the contextual order witness had already fallen to chance while the capability-only baseline remained above chance.

The result argues against treating “longer common training” as a monotone or sufficient washout strategy.

## Claim boundary for reviewers

### Supported now

- There is a reproducible controlled mechanism by which sequential training order is encoded in antisymmetric interaction terms.
- The mechanism survives a small causal transformer and language-model loss.
- Finite stage maps extend reconstruction beyond one-step local geometry.
- Exact finite pair interactions outperform the micro HVP and differential macro decoders in the fixed long-stage stress test.
- The project actively preserves and acts on negative/confounded Pythia evidence rather than treating raw classification accuracy as provenance evidence.

### Not supported yet

- General training-history recovery for realistic LLM pipelines.
- Seed-held-out chronology recovery after ordinary endpoint capabilities are matched at Pythia scale.
- Robustness to Adam state, momentum, stochastic data order, unknown stages, dataset uncertainty, distillation, merging, pruning, quantization, or later continued training.
- Black-box chronology reconstruction from generation APIs alone.
- Legal or ownership provenance claims.

## Next fixed gate

The current research decision is to isolate **model scale** before adding any other complication.

The next scale experiment should use:

- a Pythia checkpoint;
- known exact candidate stages;
- deterministic stage execution;
- plain SGD without momentum or weight decay;
- full endpoint weights;
- exact singleton and ordered-pair probes;
- a predeclared small stage-length sweep;
- stability-only learning-rate selection based on finite loss/gradient/displacement criteria, never chronology decoding accuracy.

Stochastic data order, persistent optimizer state, low-dimensional projection, unknown recipes, and black-box access are intentionally deferred until this gate is resolved.

## Reproducibility anchors

The main CI workflow executes:

```text
ruff check .
pytest
python scripts/commutator_smoke.py
python scripts/transformer_commutator_smoke.py
python scripts/operator_commutator_smoke.py
python scripts/finite_pair_commutator_smoke.py
python scripts/smoke_mvp.py
```

The current main commit prior to this reviewer-facing documentation pass is `712de4732a89f1f58d0f7ced5d81e53ae043e19b`; its CI run `33214553076` completed successfully.

Per-experiment workflow/run identifiers and frozen protocol identifiers are preserved in the individual result reports.
