# Preregistrations

Fifteen preregistration documents, a lock ledger, and an archive note. **Twelve are
content-locked** and verify against `LOCKS.jsonl`:

```bash
python src/prereg_lock.py verify --all      # expect 12 OK
```

A lock records the sha256 of the document's bytes at the moment it was locked, so a
preregistration cannot be edited after the fact without the check failing. The ledger stores the
absolute path in use at lock time — a cluster scratch path — so the verifier falls back to
matching on filename; the integrity guarantee comes from the hash, not the path.

## Disposition of every file

Several preregistrations cover lines that were **explored and then cut**. They are retained
deliberately. Deleting the preregistration of an experiment that did not make the paper is the
thing that looks like cherry-picking; keeping it, with a plain statement of what happened, is
the opposite.

| file | locked | in the paper? | disposition |
|---|---|---|---|
| `PREREG_full_cohort_causal_census.md` | ✓ | **yes** | Realized as the frozen 22-model singleton causal census (main causal figure; Supplementary S1–S3). |
| `PREREG_E11_scale_ladder.md` | ✓ | **yes** | Realized as the structural panel across the model ladder (structural figure, panel B). |
| `PREREG_exact_operator_dimensionality.md` | ✓ | **yes** | Realized as the exact bilinear-operator metrics — q₁, PR_spec, ‖U_k‖_F — replacing the diagonal proxy (structural figure, panels B and C). |
| `PREREG_encoder_decoder_dimensionality.md` | ✓ | **yes** | Realized as the encoder arm of the structural panel (harness E8; supplies the MosaicBERT and ModernBERT q₁ values). |
| `PREREG_mechanistic_tomography_E9.md` | ✓ | **yes** | Realized as the DNABERT-2 finite-intervention tomography, observer families F0–F3 (Supplementary S6). |
| `PREREG_E10_nlp_architecture_causal.md` | ✓ | superseded | Superseded in place by `_v2` below, one day later. Retained so the revision is visible. |
| `PREREG_E10_nlp_architecture_causal_v2.md` | ✓ | **yes** | Realized as the decoder/encoder causal audit (supplementary structural-detail figure, panel B). |
| `PREREG_E10b_phi3_tomography.md` | ✓ | **yes** | Realized as the Phi-3 six-row basis (supplementary structural-detail figure, panel A; Supplementary Note S1). The paper states explicitly that this result is **not** used to support any cohort claim. |
| `PREREG_E12_generator_degradation_control.md` | ✓ | **yes** | Realized as the GENERator BOS-mediation and position-rescue analysis (GENERator mechanism figure). |
| `PREREG_evo1_broadcast.md` | ✓ | no | Explored; not included. The Evo1 broadcast line was cut. Preregistration retained for transparency. |
| `PREREG_evo1_broadcast_v2_source.md` | — | no | Working source draft of the above; never locked, never realized. Retained with its parent. |
| `PREREG_cross_geometry_stageA.md` | ✓ | no | Explored; not included. Superseded by the exact-operator metric, which measures the same geometry without the diagonal approximation. |
| `PREREG_dimensionality_gate0.md` | ✓ | no | Explored; not included. Superseded by the exact-operator metric. Its `dimensionality_lib.py` survives as a dependency of three audit builders — see [`experiments/E5_dimensionality/README.md`](../../experiments/E5_dimensionality/README.md). |
| `PREREG_steering.md` | — | no | Explored; not included — **superseded by the random-direction control**, which showed the compositional phenotype tracks damage magnitude and does not require the learned row direction. The control is reported in the paper (random-direction supplementary figure); the steering line it supersedes is not. |
| `PREREG_nlp_prospective.md` | — | no | ⚠ **Disposition pending author confirmation.** Reads as the prospective NLP-cohort plan that the 22-model census later realized. Label it either "realized as the causal census" or "partially realized; remaining items not pursued" — see the note below. |
| `LOCKS.jsonl` | — | n/a | The lock ledger: one append-only record per lock, with sha256, UTC timestamp and git commit. |
| `archive/README.md` | — | n/a | Note on the superseded drafts kept under `archive/`. |

## Outstanding

`PREREG_nlp_prospective.md` is the only file whose disposition could not be determined from the
repository. It is unlocked and cites no artifact that survives, which is consistent with either
reading. **This is flagged for the authors rather than guessed**; the file is retained either
way, since an un-inferable preregistration should never be silently removed.

## A note on paths inside these documents

The preregistrations record what was planned, including repository paths that predate later
directory renames. They were deliberately **never rewritten** to match the current layout: a
preregistration is a historical record, and editing one to look current would defeat its
purpose and break its lock. Expect stale paths inside them; that is correct.
