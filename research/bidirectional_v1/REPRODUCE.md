# Reproducing the paper's bidirectional study

The native repository provides the engine, native tests, theorem note, summary, and current article. **Supplement S1 contains the complete frozen experimental workspace**, including earlier-method dependencies, all drivers, dependency pins, locks, and raw reference-distance arrays. These are not all native-root scripts.

S1 original filename: `ChronoTrace_Measurable_Value_Research_2026-09-12.zip`.
S1 SHA-256: `b70924ce131fe1545b026957ef970ae69f7442b1f4fec3f9230960f503c52412`.
Extracted workspace: `ChronoTrace_Measurable_Value_2026-09-12/`.

The submission package includes an identical copy named `Supplement_S1.zip`; the name changes, bytes do not. No permanent archive DOI is claimed. The rights holder must approve the archive and software licensing before public archival release.

```bash
cd ChronoTrace_Measurable_Value_2026-09-12
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-bidirectional.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:scripts
python scripts/verify_bidirectional_artifacts.py --root .
python -m pytest -q tests
python -O -m pytest -q tests
python scripts/run_bidirectional_confirmation.py \
  --config configs/bidirectional_v1.lock.json \
  --output artifacts/independent_rerun --workers 1
```

The first verification reads stored distances; the final command independently recomputes them. It rejects altered source hashes, dataset bytes, dependency versions, or an existing output directory. Record hardware and BLAS configuration before benchmarking runtime. See S1 README for neural bridge and serial timing commands.

The dataset is the 1,797-example scikit-learn digits subset, not the entire original UCI collection. Protocols distinguish FP64 training from FP32/FP16 observation export. The primary design uses 12 target permutations repeated across task/duration conditions; paired exports are not independent targets.
