# HVP_FEASIBILITY.md — E9 Phase 10

Written after the finite-response result is frozen (F0-F3 fit, mechanical decision
PAIR_TERMS_REQUIRED at both epsilons for DNABERT-2; GENERator scoped to dose-response).
E9 is a finite-forward-intervention study; this is a feasibility note only, not a new
experiment.

**Can the declared component directions be differentiated cleanly?** Yes in principle —
each basis component is a single down_proj output row, a standard linear readout, so a
directional derivative (HVP direction = the row's one-hot basis vector in weight space) is
well-defined and cheap to construct. No architectural obstacle.

**Could designed HVPs recover the same local pair structure more cheaply?** Unclear that
"more cheaply" would hold here. The finite measurement that already exists (256 forward
passes, ~155s total wall time, no backward pass) is already very cheap for DNABERT-2
(117M params). An HVP-based estimate of the same Gamma_ij would require a Hessian-vector
product per pair direction, i.e. a backward-mode double-differentiation through the same
model — plausible per-call cost is comparable to or higher than the finite forward call it
would replace, with no batching efficiency gained (finite masks already batch trivially
across the fixed 16-batch window set, exactly as done here). No practical speed argument
favors HVPs at this basis size (`n_D=10`, 45 pairs).

**Would the local curvature be expected to predict the finite epsilon=0.5/1.0 response?**
Not obviously. HVP-based curvature is a local (first/second-order, small-perturbation)
approximation around the unablated weights. E9's frozen scales are large, discrete
interventions (50% and 100% suppression of a full output row) — well outside the regime
where a local quadratic (Hessian) approximation is expected to hold, especially at
epsilon=1.0 (full ablation, not a small perturbation by construction). The measured
F2-vs-F3 relative MAE improvement (55% at epsilon=0.5, 24% at epsilon=1.0) already shows
the response is not even well-approximated by a *linear* (first-order) model in this
regime; there is no reason to expect a second-order Taylor expansion around the unablated
point to fare better at the *larger* of the two scales, where the true damage is largest
and least local.

**Can the full Hessian target be avoided?** Yes — this is exactly what E9 already does
(finite forward interventions only, no gradients, no HVPs, per Phase 2's explicit
instruction). Nothing in the results above motivates revisiting that choice.

**Conclusion**: finite-response recovery was neither prohibitively expensive (155s for the
full 256-condition DNABERT-2 sweep) nor scientifically ambiguous (a clean, bootstrap-backed
PAIR_TERMS_REQUIRED result at both scales). Per Phase 10's own criterion, HVP tomography is
**not** motivated as a follow-up experiment for this basis/scale combination. It might
become relevant only for a much larger basis (where forward enumeration of all masks/pairs
becomes the bottleneck) or a small-perturbation regime E9 does not study.
