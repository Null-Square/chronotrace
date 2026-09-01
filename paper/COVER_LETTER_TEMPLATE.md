# ChronoTrace Cover Letter Template

> Replace bracketed fields only after selecting the journal. Do not make stronger claims than the manuscript.

Dear Editor of **[Journal Name]**,

We submit the manuscript **“ChronoTrace: Certified Reconstruction of Training Order from Noncommutative Learning Interactions”** for consideration as a **[article type]**.

The manuscript studies an inverse problem in sequential machine learning: given a known base checkpoint and known candidate training stages, what can be certified about the unknown order in which those stages were applied to produce a finished model? This question differs from forward curriculum/order optimization, membership inference, training-data attribution, and model-lineage verification.

ChronoTrace develops a proof-oriented certificate framework based on exact ordered interactions, a witness bank frozen before higher-order candidate output, norm-safe multi-witness combinations, and proof-safe local-order linear programs. The method returns precedence relations only when a conservative lower bound excludes one orientation class; otherwise it abstains. We also establish a finite-query information barrier showing why exact fixed-depth certification for longer histories requires additional control of omitted higher-order interactions.

The empirical program preserves both negative and positive results. A preregistered single-witness higher-order certificate fails despite nonzero exact class separation, motivating the final multi-witness formulation. After freezing a label-blind two-sided decision rule, we evaluate it on a new deterministic Pythia-14M confirmation suite: the method certifies **27 of 32 complete four-stage histories** and **182 of 192 pairwise precedence relations**, with five conservative full-history abstentions and **zero contradictory certified pair relations**. Every terminal certificate is independently checked against the exact permutation-class convex hull.

We believe the paper is a good fit for **[Journal Name]** because **[2–3 sentences linking the journal’s scope to certified ML methodology / learning dynamics / model auditing]**.

The current result uses a replay-capable white-box access regime and terminal interaction depth `K=N=4`; the manuscript states these boundaries explicitly and does not claim black-box provenance or universal polynomial-time exact reconstruction.

The manuscript is original, is not under review elsewhere, and **[preprint statement if applicable]**. All authors have approved the submission. **[Conflict/funding/data/code statements as required.]**

Thank you for your consideration.

Sincerely,

**[Corresponding author]**  
**[Affiliation]**  
**[Email]**
