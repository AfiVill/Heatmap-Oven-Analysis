# Methodology Notes

A closer look at two engineering decisions from this project, written up because the reasoning behind them generalizes beyond this specific application.

## Diagnosing a model-evaluation problem without touching a hyperparameter first

Early model evaluations produced R² scores that were low, and occasionally negative — a red flag, but not by itself a diagnosis. The instinct in that situation is often to start turning knobs: more epochs, a different learning rate, a bigger model. That approach can accidentally "fix" a symptom while leaving the actual cause in place, which tends to resurface later in a more confusing form.

Instead, the investigation worked backward from the metric itself:

1. **Confirm the model and the math are correct in isolation.** Before questioning the data, the architecture and loss computation were verified independently — ruling out a correctness bug as the explanation.
2. **Characterize the target distribution.** R² is a ratio against the *variance of the target*. A low-variance target makes R² an unforgiving metric even when absolute prediction error (RMSE) is small — worth checking before concluding the model itself is failing.
3. **Look at what the input actually encodes.** The most consequential finding: two physically distinct output surfaces were being predicted from one shared input representation, with nothing in that input distinguishing which surface was being asked for. That's not a training bug — it's a modeling decision with a hard, computable ceiling on achievable accuracy, and it had been made implicitly rather than deliberately.
4. **Quantify the effective sample size**, accounting for correlation between samples drawn from the same physical experiment run, rather than trusting the raw row count.

Each of these was a testable claim, checked against real data before being accepted — not a hypothesis adopted because it sounded plausible. The output of this process wasn't a tuned model; it was a short, ranked list of structural causes, each with a proposed fix and an expected effect size, handed back for a decision on which to act on first.

## Adding a feature without breaking what already works

A recurring risk in ML systems that are already in use: an improvement to the model input shape breaks every checkpoint trained before the change. The thermal-exposure input channel described in the main README was designed against that constraint explicitly:

- The channel is **optional at the data level** — pair objects either carry it or they don't, and the training/inference code paths detect which case they're in rather than assuming one.
- The channel count that a checkpoint expects is **read from the checkpoint's own saved weights**, not hardcoded — so older and newer checkpoints load correctly side by side without a version flag.
- Combinations that would silently produce an unusable checkpoint (for example, pairing this feature with an architecture whose input layer can't accommodate it) are **rejected at training time with a specific error**, rather than allowed to train "successfully" into a model with no valid inference path.

None of this is exotic — it's the ordinary discipline of not trading a working system's stability for a new feature's convenience.
