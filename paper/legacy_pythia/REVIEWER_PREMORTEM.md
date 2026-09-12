# Adversarial Reviewer Premortem

This is an internal pre-submission review. The goal is to answer likely rejection arguments in the manuscript itself, not after review.

## R1. “This is just another demonstration that training order matters.”

**Answer:** No. The paper explicitly cites forward curriculum/data-order and Lie-bracket work. ChronoTrace asks the inverse question: the chronology is hidden, candidate stage identities are known, and the output is a certified precedence relation or abstention. Keep this distinction in the first page and related-work table.

## R2. “The mathematics is standard Möbius inversion plus Sherali--Adams.”

**Answer:** The paper does not claim those ingredients individually. The contribution is the inverse-chronology formulation and certificate architecture: ordered endpoint interactions, witnesses frozen before higher-order output, L1-safe aggregation, chronology-property optimization, corrected dual certification, and label-blind two-sided decisions. The empirical protocol validates that architecture end-to-end.

## R3. “Why not enumerate all 24 histories?”

**Answer:** We do, as a spent mechanism result, and it recovers 24/24. That establishes endpoint identifiability but costs full-history enumeration. ChronoTrace instead organizes evidence by interaction degree and property classes. At fixed K the representation is polynomial-size, while the paper explicitly does not claim universal exact fixed-depth reconstruction.

## R4. “Your final N=4,K=4 result is also terminal/factorial.”

**Answer:** Correct; this is a central limitation, not hidden. The final experiment validates certificate correctness and generalization of the frozen decision rule. The information-barrier theorem explains why exact fixed-depth scalability is nontrivial. The paper claims an exact terminal level plus a polynomial-size fixed-depth hierarchy—not an arbitrary-N exact decoder.

## R5. “The method was tuned on the same instance.”

**Answer:** Development was performed on a spent ABCD instance and labeled accordingly. The preregistered single-witness K4 result remained negative. The multi-witness method was developed post-hoc, then made label-blind and frozen. Final confirmation used a new deterministic four-seed set after discovering that the original reserved v1 seeds had already been consumed. No intermediate adaptation was allowed.

## R6. “The held-out provenance is questionable because you changed seeds.”

**Answer:** The seed change strengthens, rather than weakens, provenance. An audit found the original four seeds had already been executed in an earlier workflow; the repository records them as spent instead of pretending they remained held-out. Final v3 seeds were mechanically SHA-256-derived before execution, and their derivation rule is frozen in the protocol.

## R7. “The aggregate failed, so the result is invalid.”

**Answer:** All four scientific jobs completed and uploaded immutable artifacts before aggregation. The aggregate failure was a JSON key-order assertion: files were emitted with sorted dictionary keys, while the aggregator compared iteration order with the target list. The correction only checks exact target key set/count. No scientific result, threshold, or method changed; a sorted-key regression test and full CI pass.

## R8. “27/32 is not high enough.”

**Answer:** The method is a certificate system, not a forced classifier. It certified 84.4% of complete histories and 94.8% of pair relations, with ten ambiguous pair decisions and zero contradictory certified pairs. Forcing labels on the five unresolved histories would increase coverage but destroy the meaning of certification. Report coverage and abstention separately.

## R9. “Zero observed contradictions does not prove zero false-positive probability.”

**Answer:** Agreed. Do not write a probabilistic universal claim. State exactly that zero contradictory certified pair relations were observed in the 192 fresh pair decisions, and that terminal certificates were independently checked against exact class geometry.

## R10. “The access regime is unrealistic.”

**Answer:** It is intentionally strong and explicitly stated: known base, candidate stages, training rule, replay access, final weights. The paper is a mechanism/certification result, not a black-box deployment claim. Future work can weaken access only after the inverse mechanism is established.

## R11. “Synthetic codebooks do not represent real post-training domains.”

**Answer:** Correct. Synthetic controlled stages permit exact deterministic replay, hash-level provenance, and geometry checks. This is a tradeoff between ecological realism and certificate auditability. The limitation is explicit; do not imply semantic-domain generalization.

## R12. “One Pythia size is too narrow.”

**Answer:** This is the strongest remaining empirical weakness. The current paper should be positioned as theory + certified controlled model-scale validation. If the chosen journal requires broad scale sweeps, that is a venue-fit/revision question, not a reason to silently retune the current method after confirmation.

## R13. “Why is the single-witness negative important?”

**Answer:** Because it establishes that exact higher-order information and Euclidean class separation are insufficient if the chosen witness direction is poor. The multi-witness step addresses a diagnosed geometric failure rather than merely adding capacity after a disappointing number.

## R14. “Could the multi-witness optimizer itself overfit the target?”

**Answer:** On the spent development instance, yes—that is why the result is post-hoc. The confirmatory method freezes the witness construction and label-blind pairwise certificate before fresh data. For every pair it tests both orientations before consulting the generating chronology.

## R15. “Why should I trust the LP lower bound numerically?”

**Answer:** The raw solver dual is not trusted. ChronoTrace recomputes a proof-safe corrected bound using reduced-cost minima and a numerical guard. At N=K=4 every orientation-class primal is independently checked against the complete-permutation convex hull; maximum terminal discrepancy is approximately `4.06e-17`.

## R16. “The information barrier is obvious because you can alter an unqueried function.”

**Answer:** Its value is not that adversarial interpolation is philosophically surprising. It makes the paper’s scalability boundary formal: without regularity or extra K+1 information, no amount of clever post-processing of the same finite low-degree observations can give a universal exact tail certificate. Present it as a boundary theorem, not as the sole novelty claim.

## R17. “Why a Q1 journal rather than a workshop?”

**Answer:** The package now contains a full inverse-problem formulation, exact identities, a proof-safe certified algorithm, a barrier theorem, falsification-driven method development, fresh confirmation, and unusually detailed reproducibility/provenance. The tradeoff is narrow empirical breadth. That combination is credible for a strong journal whose scope values methodology/theory/auditing rather than benchmark scale alone.

## Final pre-submission pass

A reviewer should be able to answer these questions from the paper without opening the repository:

1. What is known, hidden, and observed?
2. What exactly is certified?
3. Why is the certificate conservative?
4. Which experiment was negative?
5. Which analysis was post-hoc?
6. Which result is fresh confirmation?
7. What does 27/32 mean versus the five abstentions?
8. Why does N=4,K=4 not imply arbitrary-N scalability?
9. How does the problem differ from forward Lie-bracket planning and provenance/lineage work?
10. Where can every run/artifact/hash be audited?

If any answer requires oral explanation, improve the manuscript before submission.