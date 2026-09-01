# Pythia-14M reproducibility and portable scale gate

Date: 2026-08-29

Status: **reproducible scientific negative for full-order finite-pair decoding; scale gate blocked**

## Why this gate existed

The first frozen Pythia-14M finite-pair chronology bridge produced contradictory outcomes under the same scientific protocol: one execution recovered `3/6` A/B/C histories and a later execution recovered `6/6`. Because the learning rate had already been selected using chronology-blind singleton stability, the contradiction was treated as a reproducibility problem rather than evidence for or against the decoder.

No Pythia-31M chronology experiment was allowed to start.

## Frozen scientific protocol

- model: `EleutherAI/pythia-14m-deduped`
- revision: `step143000`
- optimizer: plain SGD, no momentum, no weight decay
- learning rate: `1e-4`
- stage length: `16` updates per stage
- stages: A, B, C
- full histories: all `3! = 6` permutations
- model precision: FP32
- deterministic full batches
- finite-pair basis: 9 stage executions
- ground-truth validation: 18 additional stage executions
- success criterion: recover all `6/6`, positive decode margin, identifiable finite-pair signatures, and non-identifiable orientation ablation

The tokenizer/codebook/data lock was unchanged. The exact base parameter vector and exact training tensors were fingerprinted before training.

## First reproducibility intervention

Three independent GitHub-hosted runners were launched with the same pinned environment:

- Python `3.11.16`
- PyTorch `2.13.0+cpu`
- Transformers `5.16.1`
- Tokenizers `0.23.1`
- huggingface-hub `1.29.0`
- safetensors `0.8.0`
- `PYTHONHASHSEED=0`
- one intra-op thread
- one inter-op thread
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `torch.use_deterministic_algorithms(True)`

The gate required exact SHA-256 agreement for the base model, training batches, finite-pair basis, candidate signatures, and all six full-history endpoints. It also required all three replicas to recover `6/6`.

Workflow: `33218688360`.

### Result of first intervention

The gate failed reproducibility.

| Replica | CPU | Correct | Outcome |
| --- | --- | ---: | --- |
| 1 | AMD EPYC | 3/6 | fail |
| 2 | AMD EPYC | 3/6 | fail |
| 3 | Intel Xeon | 6/6 | pass |

The aggregate comparator reported mismatches in the finite-pair basis, candidate signatures, and full-history endpoints, plus inconsistent decoded chronologies.

Crucially, all replicas had the same scientific protocol fingerprint, base parameter hash, tokenizer/codebook/data identity, exact stage-batch tensor hashes, software versions, PyTorch build hash, one-thread settings, and deterministic-algorithm flag. The two AMD replicas also differed from each other, so CPU model name alone was not a sufficient explanation.

**Interpretation:** no Pythia-scale positive result was accepted. The first divergence occurred inside the learned finite-stage operator.

## Portable numerical intervention

A second adjudication changed **only numerical execution controls**, not scientific hyperparameters.

Frozen additions:

- `ATEN_CPU_CAPABILITY=default`
- `MKL_CBWR=COMPATIBLE`
- `MKL_DYNAMIC=FALSE`
- `OMP_DYNAMIC=FALSE`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- MKLDNN disabled before model execution
- one PyTorch intra-op thread
- one PyTorch inter-op thread
- `torch.use_deterministic_algorithms(True)`

The runner hard-refused execution if the numerical-control environment did not match the lock.

Workflow: `33219286064`.

## Final portable result

The portable run is **exactly reproducible across all three replicas**.

The three replicas ran on different hosted CPU models, including AMD EPYC 7763 and AMD EPYC 9V74, but produced the same:

- scientific fingerprint: `73f8f88b3ddb8ccd6b89791d597067f33da22582496c45cd92c28c11a0416f90`
- numerical execution fingerprint: `deaad55af513e78d0c4c1d5636836bcb7d7325be64d8df0b196cd6a66b262d42`
- base parameter hash: `cba585ef12f0a770686bffb9d1c1d00e11400106b46d943c1cae04fa7e0df2ce`
- finite-pair basis hash: `1afeaa53d3b98c32473fcdfc50c297e6f2db14226d9a5f868ef3aa5366f882c2`
- bundle hash for all six full-history endpoints: `ca3e24c4139d87ec4004e78c26c229aa0cf00d86956c5ffd8b4163469622a9c7`
- decoded result: **`3/6`**

The comparator failed only because every replica failed the frozen `6/6` scientific criterion. It did **not** report tensor-fingerprint disagreement.

### Aggregate finite-pair diagnostics

- correct histories: `3/6`
- accuracy: `0.500`
- minimum finite-pair signature separation: `0.2056144774`
- minimum decode margin: `0.0150282830`
- maximum triple+ remainder norm: `0.2054421455`
- maximum `2 * ||r_high|| / delta_min`: `1.9983237374`
- orientation ablation identifiable: **false**
- orientation-ablation signature separation: `0.0`

Singleton displacement norms:

- A: `0.2062330991`
- B: `0.1697814167`
- C: `0.1847849786`

### Exact portable chronology result

| Actual history | Decoded | Correct | Margin | Higher-order ratio |
| --- | --- | :---: | ---: | ---: |
| `ABC` | `ACB` | no | `0.0150283` | `1.69037` |
| `ACB` | `ACB` | yes | `0.0198729` | `1.52104` |
| `BAC` | `BAC` | yes | `0.0323356` | `1.69944` |
| `BCA` | `BAC` | no | `0.0304868` | `1.99832` |
| `CAB` | `CAB` | yes | `0.0423949` | `1.56819` |
| `CBA` | `CAB` | no | `0.0374913` | `1.98057` |

The three wrong predictions are not arbitrary. Each preserves the true first stage and swaps only positions 2 and 3.

Descriptive partial-order diagnostics on this single locked synthetic instance:

- first-stage identity: `6/6`
- correct pairwise precedence relations: `15/18 = 83.3%`
- mean Kendall tau between true and decoded permutations: `2/3`

These diagnostics were noticed after the full-order failure and therefore are **hypothesis-generating**, not confirmatory evidence. They must be tested on independently generated stage worlds / codebooks before being claimed.

## Scientific interpretation

The portable result resolves the ambiguity:

> The base-checkpoint finite-pair truncation is insufficient to reconstruct the complete three-stage Pythia-14M chronology under the frozen 16-update protocol.

This is a real negative result for the specific full-order pairwise decoder.

It does **not** show that the endpoint contains no chronology information. The structured adjacent-swap errors, nonzero directed-pair signature separation, and 15/18 descriptive precedence accuracy motivate a sharper mechanism.

For a finite stage map `F_D`, define the state-dependent pair commutator

`C_jk(theta) = F_k(F_j(theta)) - F_j(F_k(theta))`.

Then the difference between histories that share a prefix A and swap the remaining B/C order is exactly

`F_C(F_B(F_A(theta0))) - F_B(F_C(F_A(theta0))) = C_BC(F_A(theta0))`.

The current finite-pair basis instead uses the B/C interaction measured at the **base checkpoint**, `C_BC(theta0)`. The difference

`C_BC(F_A(theta0)) - C_BC(theta0)`

is a prefix-conditioned third-order effect. This offers a direct mechanism for the observed pattern: pair interactions involving the earliest stage are measured near the state where they actually act, while the interaction between stages 2 and 3 can drift after the first stage changes the model.

The next experiment should therefore test **state-conditioned interaction order**, not retune the existing pairwise decoder.

## Scale decision

Pythia-31M chronology remains blocked.

Before scaling, the project must:

1. formalize the interaction hierarchy and prefix-conditioned commutator theory;
2. freeze independent 14M stage/data seeds before testing the newly observed first-stage / pairwise-precedence hypothesis;
3. measure whether prefix-conditioned third-order information explains and resolves the tail-order ambiguity;
4. keep full-order recovery, pairwise precedence, first-stage recovery, and higher-order residual as separate reported endpoints.
