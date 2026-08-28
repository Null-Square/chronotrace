# Pythia-14M reproducibility gate

Date: 2026-08-29

Status: **negative result / scale gate blocked**

## Why this gate existed

The first frozen Pythia-14M finite-pair chronology bridge produced contradictory outcomes under the same scientific protocol: one run recovered `3/6` A/B/C histories and a later run recovered `6/6`. Because the learning rate had already been selected using chronology-blind singleton stability, the contradiction was treated as a reproducibility problem rather than evidence for or against the decoder.

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

## Reproducibility intervention

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

## Result

The gate failed.

| Replica | CPU | Correct | Basis hash prefix | Outcome |
| --- | --- | ---: | --- | --- |
| 1 | AMD EPYC 9V74 | 3/6 | `8f036a23...` | fail |
| 2 | AMD EPYC 9V74 | 3/6 | `06b503c7...` | fail |
| 3 | Intel Xeon 6973P-C | 6/6 | `7db84fcd...` | pass |

The aggregate comparator reported mismatches in the finite-pair basis, candidate signatures, and full-history endpoints, plus inconsistent decoded chronologies.

Crucially, all replicas had the same:

- scientific protocol fingerprint: `73f8f88b3ddb8ccd6b89791d597067f33da22582496c45cd92c28c11a0416f90`
- base parameter hash: `cba585ef12f0a770686bffb9d1c1d00e11400106b46d943c1cae04fa7e0df2ce`
- tokenizer fingerprint and codebook hash
- stage A/B/C data hashes
- exact stage-batch tensor hashes
- Python/PyTorch/Transformers/tokenizer versions
- PyTorch build-configuration hash
- one-thread settings
- deterministic-algorithm flag

The two AMD replicas also differed from each other substantially, so CPU model name alone does not explain the divergence.

### Replica 1

Singleton displacement norms:

- A: `0.1664831`
- B: `0.2116965`
- C: `0.2160951`

Decoded histories:

- `ABC -> ACB`
- `ACB -> ACB`
- `BAC -> BCA`
- `BCA -> BCA`
- `CAB -> CBA`
- `CBA -> CBA`

Minimum signature separation: `0.2330237`.
Minimum decode margin: `0.0015561`.
Maximum higher-order ratio: `1.81098`.

### Replica 2

Singleton displacement norms:

- A: `0.2164339`
- B: `0.1595301`
- C: `0.1991630`

Decoded histories:

- `ABC -> ACB`
- `ACB -> ACB`
- `BAC -> BAC`
- `BCA -> BAC`
- `CAB -> CAB`
- `CBA -> CAB`

Minimum signature separation: `0.1870460`.
Minimum decode margin: `0.0169723`.
Maximum higher-order ratio: `2.27308`.

### Replica 3

Singleton displacement norms:

- A: `0.0907289`
- B: `0.0890678`
- C: `0.0897069`

All six histories were recovered correctly.
Minimum signature separation: `0.0559330`.
Minimum decode margin: `0.0045323`.
Maximum higher-order ratio: `1.92108`.

## Interpretation

The first divergence occurs inside the learned finite-stage operator, before chronology decoding. This rules out the base checkpoint, tokenizer, generated data, concrete token batches, software package versions, thread count, and chronology-selection logic as sufficient explanations.

The remaining leading hypothesis is host-dependent CPU numerical dispatch/backend behavior. PyTorch CPU kernels can select different ISA-specific implementations at runtime. The current run also had MKLDNN enabled. The observed difference is large enough to change the finite-pair basis and chronology ranking, not merely the final printed floating-point digits.

This is therefore **not a successful Pythia-scale ChronoTrace result**. It is a failed reproducibility gate.

## Next gate

Keep every scientific hyperparameter above unchanged and test a portable CPU numerical path:

1. force `ATEN_CPU_CAPABILITY=default`, which PyTorch documents as selecting the oldest supported CPU vector instruction path;
2. disable MKLDNN before model execution;
3. retain the pinned package versions and single-thread settings;
4. run three independent replicas again;
5. require exact tensor-hash agreement and `6/6` recovery before any Pythia-31M chronology compute.

The portable-kernel settings are chosen to remove the identified numerical-dispatch variable. They are not selected using chronology accuracy.
