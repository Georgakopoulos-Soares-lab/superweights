# Claude Code Prompt — Build the Manuscript Source-of-Truth Package

We are now preparing the manuscript for the **Superweights / high-gain gated-FFN project**. Do **not** write the manuscript yourself.

Your task is to inspect the full repository and produce a **manuscript source-of-truth package** that I can give to ChatGPT to write the paper accurately, without guessing, hallucinating methods, or accidentally using retired results.

Treat the repository artifacts as authoritative. Prefer raw machine-readable evidence (`JSON`, `CSV`, `NPZ`, logs, scripts, preregistration files) over prose summaries or old Markdown notes. If two sources conflict, explicitly report the conflict rather than choosing the more convenient result.

The current conceptual framing is:

> High-gain gated-FFN structures recur across text and genomic foundation models, but structural prominence, functional criticality, and finite-intervention causal response complexity are distinct quantities. Encoder/decoder architecture organizes structural geometry more consistently than it organizes causal mechanism.

Keep these concepts separate:

`STRUCTURAL GEOMETRY != FUNCTIONAL CRITICALITY != CAUSAL RESPONSE COMPLEXITY`

Important known constraints:

- Use exact `||U_k||_F` as structural magnitude.
- `q1` measures spectral concentration/rank-1-ness and must **not** be treated as magnitude or causal importance.
- Do not resurrect retired/buggy NTv3 results.
- Do not treat the old large DNABERT splice/codominance effects as established unless you find reconciled raw evidence.
- Do not claim that the GENERator high-gain row causes the BOS attention sink.
- Do not claim encoder = interactional and decoder = additive; Phi-3 disproves that.
- Do not add new exploratory experiments. This task is evidence collection and manuscript preparation only.

Please produce one main file:

`MANUSCRIPT_SOURCE_OF_TRUTH.md`

You may additionally create CSV/TSV files for large tables if useful.

## 1. Paper framing and outline

Provide:

- 3–5 defensible title options, if the repository does not already contain final ones.
- A concise statement of the paper's central claim.
- The final recommended four-part Results structure.
- What each Results section establishes and what it explicitly does **not** establish.
- Recommended Discussion-level takeaways.

The expected broad structure is:

1. Structural organization across text and genomic models
2. Causal organization is heterogeneous despite structural trends
3. DNABERT-2 mechanistic deep dive
4. GENERator functional / biological deep dive

If the repository supports a better finalized organization, report it and explain why.

## 2. Final figure map

For every main and supplementary figure, provide:

- figure number
- panel letter
- purpose of panel
- exact artifact/file supplying the plotted data
- exact metric
- model(s)
- sample size / seeds / prompts / sequences / intervention count as applicable
- key numeric result
- safe manuscript interpretation
- claims that the panel must **not** be used to make

Include the final intended Figure 1–4 structure and note whether the latest figure-generation scripts already reflect it.

Also provide draft figure legends based only on supported evidence.

## 3. Headline results table

Create a compact table containing every quantitative result likely to appear in the Abstract, Results, or Discussion.

For each result include:

- model
- experiment
- metric
- exact value
- uncertainty / CI / p-value if available
- units
- N
- number of seeds
- intervention strength
- baseline/control
- artifact path
- script that generated or analyzed it
- whether it is `MAIN-TEXT SAFE`, `SUPPLEMENT ONLY`, `PROVISIONAL`, or `RETIRED`

Include at minimum the raw-backed results for:

### Structural analysis

- exact `q1` / `PR_spec`
- exact or layer-relative `||U_k||_F`
- encoder vs decoder structural comparison
- same-layer ordinary-row controls

### E10 decoder causal audit

- Llama
- Mistral
- OLMo
- Phi-3
- Qwen2.5
- K values
- singleton causal effects
- OLMo structural-vs-causal ranking dissociation

### Encoder tomography

- DNABERT-2
- MosaicBERT
- ModernBERT
- F0/F1/F2/F3 metrics
- F2→F3 improvements and bootstrap intervals

### Phi-3 E10b

- epsilon 0.5 results
- epsilon 1.0 results
- pair-term decision
- unresolved full-ablation regime
- descriptive three-layer-2-row pattern, clearly labelled descriptive rather than fitted higher-order mechanism

### DNABERT-2

- pretrained L9/r264 + L9/r294 interaction
- E9 epistasis
- observer-ladder metrics
- known pair ranking #1/45
- any safe additional raw-backed mechanistic numbers

### GENERator

- GC intervention values for row 2371
- secondary row 1522
- random-control comparison
- BOS attention share
- fold over uniform
- fraction of heads argmaxing position 0
- BOS high-gain activation magnitude

### NTv3

Only include a result if the corrected raw artifact genuinely exists. Otherwise state that the result cannot currently support a plotted/main quantitative claim.

## 4. Exact methods needed for manuscript writing

For every experiment appearing in the proposed main text, extract enough information to write a reproducible Methods section.

### Models

Provide:

- exact Hugging Face / repository model name
- revision / commit hash where available
- parameter scale
- encoder or decoder
- architecture family
- relevant FFN implementation
- exact module paths (`down_proj`, `gate_proj`, `up_proj`, etc.)
- tokenizer details if relevant

### Structural analysis

Give the exact mathematical definitions used for:

- `U_k`
- exact `||U_k||_F`
- Gram/Hadamard construction
- `q1`
- `PR_spec`
- layer-relative magnitude
- candidate-row selection
- ordinary-row controls

Explain how the earlier diagonal approximation differed and confirm it is not used for final claims.

### Causal interventions

For each causal experiment provide:

- what parameter/object is intervened on
- mathematical intervention definition
- epsilon / alpha meaning
- how ablation differs from partial suppression
- output metric
- baseline
- random / neighboring / other controls
- number of inputs
- sequence length / tokenization
- aggregation procedure
- statistical procedure

### Tomography

Define exactly:

- frozen basis construction
- F0
- F1
- F2
- F3
- training/held-out split
- feature matrix
- pair terms
- fit procedure
- R²
- MAE
- relative MAE improvement
- bootstrap CI
- decision rule for `PAIR_TERMS_REQUIRED`
- any adequacy thresholds

### DNABERT-2

Provide:

- exact pretrained MLM evaluation setup
- sequences/data
- masking/evaluation protocol
- interaction / epistasis definition
- E9 basis construction
- intervention strengths
- held-out evaluation procedure

If downstream splice or codominance experiments are discussed anywhere, clearly separate:

1. old unsupported/prose-only values
2. new raw-backed reproductions
3. whether they belong in the paper at all

### GENERator

Provide:

- exact model/checkpoint
- prompt/sequence generation setup
- number and length of generations
- GC calculation
- alpha intervention
- random controls
- BOS attention extraction
- which layers/heads were included
- activation extraction method
- how "uniform attention" baseline was computed

## 5. Dataset and corpus inventory

For every dataset/corpus/task used in a main or supplement experiment, provide:

- name
- source
- split
- sample count
- filtering
- preprocessing
- random seed
- sequence/window length
- tokenizer
- whether labels are used
- citation if known

Do not infer missing details.

## 6. Provenance and preregistration

Build a provenance table for all headline claims containing:

- claim
- artifact path
- generating script
- analysis script
- preregistration / decision ID
- checkpoint/model revision
- status

Explicitly identify:

- preregistration versions
- the E10 q1-selection mistake and corrected prereg v2
- which causal measurements were performed only after correction
- retired or superseded artifacts
- any result still backed only by prose rather than raw data

## 7. Retired, corrected, disputed, or unsafe claims

Create a dedicated section listing every result that the manuscript writer should avoid or qualify.

At minimum inspect:

- old DNABERT catastrophic splice pair magnitude
- old DNABERT codominance magnitude
- NTv3 truncation bug
- old approximate `U_k` analysis
- Evo1 numerical saturation issue
- any retracted NTv3 / Evo1 mechanism claims
- any structural/systemic decomposition that no longer survives
- any claim that q1 predicts causal importance
- any claim that architecture fully determines causal organization
- any claim of GENERator causal control over the BOS attention sink
- any implied continuous/linear GC steering claim

For each, say exactly why it is unsafe and whether there is a corrected replacement.

## 8. Controls and falsification evidence

Summarize the controls supporting the paper:

- random-row controls
- same-layer ordinary-row controls
- neighboring controls if used
- noise-floor/determinism checks
- exact-operator correction
- held-out tomography
- bootstrap intervals
- independent architecture replications
- structural-positive / causal-negative examples

State what each control rules out and what it does not rule out.

## 9. Limitations

Provide an evidence-grounded manuscript-ready limitations inventory.

In particular consider:

- limited number of architectures
- incomplete structural/causal candidate coverage for K=1 models
- finite intervention strengths
- tomography only up to second order
- Phi-3 failure at full ablation
- no causal proof linking GENERator BOS attention sink to the high-gain pathway
- disputed downstream DNABERT biological phenotype magnitudes
- structural measures do not predict causal importance universally
- architecture association is descriptive rather than a universal law

Do not invent limitations unsupported by the project design.

## 10. References

Extract a clean citation inventory for works actually needed by the manuscript.

For each provide, if available:

- citation key
- authors
- title
- venue/year
- DOI/arXiv/URL
- what manuscript claim it supports

Especially identify literature for:

- Yu et al. super weights
- massive activations / attention sinks
- transformer/GLU/SwiGLU mechanisms if used
- each genomic foundation model
- any interpretability/causal-analysis methodology directly cited

Flag references that exist in notes but cannot be verified from the repository.

Do not fabricate bibliographic metadata.

## 11. Manuscript logistics

Report any repository evidence for:

- intended venue
- formatting requirements
- abstract word limit
- section requirements
- figure limits
- author list
- affiliations
- corresponding author
- acknowledgments
- funding
- code/data availability
- author contributions
- competing interests

If unknown, write `UNKNOWN — USER INPUT REQUIRED`.

## 12. Unresolved manuscript blockers

End with a section titled:

`## UNRESOLVED FACTS FOR CHATGPT`

List only facts that genuinely cannot be determined from the repository but would be necessary for manuscript writing.

Separate them into:

- manuscript-blocking
- can safely remain as placeholders
- optional/nonessential

Do **not** ask me questions during the audit. Finish the repository inspection first and report what is actually missing.

## Output requirements

The package must be detailed enough that another model can write the complete manuscript **without inspecting the repository itself**.

Every important number should have a traceable source.

When evidence is ambiguous, say so.

When repository prose disagrees with raw results, show both and recommend the raw-backed interpretation.

Do not write polished manuscript prose except for concise claim statements and figure legends.

Do not run new scientific experiments.

Do not "improve" the story by adding unsupported claims.

At the very end, provide a short checklist confirming whether the package contains enough evidence to write:

- Title
- Abstract
- Introduction
- Results 1
- Results 2
- Results 3
- Results 4
- Discussion
- Methods
- Figure legends
- References
- Supplementary notes

For each mark `READY`, `READY WITH PLACEHOLDER`, or `BLOCKED`, and explain any `BLOCKED` item in one sentence.
