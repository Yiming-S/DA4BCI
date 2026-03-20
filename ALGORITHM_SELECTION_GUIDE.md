# Algorithm Selection Guide for DA4BCI

This guide is meant to help you choose a domain adaptation method in `DA4BCI`
based on the type of shift you expect, the way your EEG data are represented,
and the amount of supervision you have. It is organized for practical use, not
by paper chronology.

## Start with the shift you expect

| Shift pattern | Typical signs | Good first choices | Escalate to |
| --- | --- | --- | --- |
| Mean or covariance shift | Source and target look similar, but one domain is offset or has a different spread | `CORAL`, `SA` | `RD`, `PT`, `ART` |
| Subspace shift | Principal directions differ across sessions or subjects | `SA`, `TCA` | `GFK`, `MIDA` |
| Nonlinear sample-geometry shift | Clusters bend, split, or move in ways that linear transforms do not capture well | `OT` | `MIDA`, `TCA` |
| Mixed shift with class structure | Both marginal and class-conditional alignment matter | `TCA`, `PT`, `ART` | `M3D` |
| Label prior shift | Class proportions change but class-conditional structure is mostly stable | `label_shift_em()` | Combine with a feature alignment method if needed |
| Covariate shift with training reweighting | Source samples are still useful but need importance weighting | `kmm_weights()` | Combine with a base classifier or a feature alignment method |

## Choose by data representation

### 1. Ordinary feature matrices

Use these methods when your data are already represented as a numeric matrix
with rows as samples and columns as features:

- `SA`: fast linear baseline for subspace mismatch.
- `CORAL`: fast baseline when second-order statistics differ.
- `TCA`: kernel-based latent alignment for marginal shift.
- `MIDA`: kernelized dependence control with domain labels.
- `GFK`: smooth interpolation between source and target subspaces.
- `OT`: sample-level geometric matching.
- `M3D`: multi-stage, class-aware refinement when source labels are available.

### 2. Covariance-aware or SPD-oriented EEG features

Use these methods when covariance structure is central to the problem, such as
cross-subject EEG adaptation where second-order geometry carries signal:

- `RD`: lightweight covariance rotation baseline.
- `PT`: strong default when SPD geometry is the main signal.
- `ART`: geometry-aware transport with shrinkage-based covariance estimation.

These methods still consume matrices in this package, but they are most useful
when those matrices summarize stable EEG structure rather than arbitrary
high-noise raw features.

### 3. Trial-level EEG structure

If your data are still stored as trials of shape channels x time, the most
useful entry point is usually not a main DA method yet. First consider:

- `euclidean_alignment()`: align trials before feature extraction, covariance
  estimation, or downstream adaptation.

In practice, a common workflow is:

1. Apply trial-level alignment if needed.
2. Extract features or covariance summaries.
3. Run a domain adaptation method on the resulting matrix representation.

## Choose by supervision

| Method | Source labels required | Uses target pseudo-labels | Notes |
| --- | --- | --- | --- |
| `SA` | No | No | Unsupervised baseline |
| `CORAL` | No | No | Unsupervised baseline |
| `TCA` | No | No | Unsupervised kernel method |
| `MIDA` | No | No | Unsupervised in this package interface |
| `GFK` | No | No | Unsupervised subspace-geometric method |
| `RD` | No | No | Unsupervised covariance alignment |
| `PT` | No | No | Unsupervised SPD transport |
| `ART` | No | No | Unsupervised SPD transport |
| `OT` | No | No | Unsupervised sample transport |
| `M3D` | Yes | Yes | Requires `source_labels` and iteratively updates target pseudo-labels |
| `label_shift_em()` | Indirectly | Yes | Adjusts priors using source-model posteriors on target data |
| `kmm_weights()` | No | No | Reweights source samples under covariate shift |

## Recommended starting points

- If you are unsure, start with `SA` and `CORAL`.
- If EEG covariance structure is the main signal, start with `PT`, then compare
  against `ART`, and keep `RD` as a lightweight baseline.
- If principal directions drift across sessions or subjects, try `SA`, then
  `TCA` or `GFK`.
- If point-to-point geometry matters more than a global linear map, try `OT`.
- If simpler methods fail and you have reliable source labels, try `M3D` last.
- Use `MIDA` when you specifically want a kernelized method and are willing to
  tune bandwidth and regularization carefully.

## Method selection matrix

| Method | Best for | Main strength | Main assumption | Failure risk | Relative cost | Recommended priority |
| --- | --- | --- | --- | --- | --- | --- |
| `SA` | Linear subspace shift on ordinary features | Fast, simple, strong baseline | Leading PCs capture the domain gap | Misses nonlinear or covariance-heavy mismatch | Low | Start here |
| `CORAL` | Mean/covariance mismatch | Matches second-order statistics with minimal tuning | Covariance shift is the dominant problem | Weak when shift is strongly nonlinear or multimodal | Low | Start here |
| `TCA` | Marginal shift with latent shared structure | Kernel latent space can reduce mismatch beyond linear alignment | A shared latent representation exists | Sensitive to `k`, `sigma`, and regularization | Medium | Early option |
| `MIDA` | Domain-dependent nuisance variation in kernel space | Flexible kernelized dependence control | Domain labels encode nuisance structure well | Tuning-sensitive; can behave poorly if bandwidth is off | Medium to high | Targeted option |
| `GFK` | Smooth subspace drift between domains | Geometric interpolation between subspaces | Source and target are linked by a useful Grassmann path | Less effective when mismatch is not mainly subspace-based | Medium | Early to mid option |
| `RD` | Quick covariance/eigenspace alignment | Lightweight geometry-aware baseline | Covariance eigenspaces explain the shift | Can be crude when class structure or nonlinear effects dominate | Low to medium | Baseline for covariance settings |
| `PT` | Cross-session or cross-subject covariance alignment | Strong SPD transport interpretation and practical default | Stable covariance estimates represent domain structure | Can degrade with poor covariance estimation or noisy features | Medium | High priority for EEG covariance |
| `ART` | SPD-geometry alignment with shrinkage | Robust covariance transport with shrinkage and geometry tools | Covariance geometry is meaningful and estimable | Can underperform if the data are not well summarized by covariances | Medium | High priority for EEG covariance |
| `OT` | Nonlinear sample geometry or multimodal mismatch | Matches samples directly rather than only moments or subspaces | The chosen transport cost reflects meaningful similarity | Sensitive to cost scaling and regularization; can over-smooth | High | Use after simple baselines |
| `M3D` | Complex mixed shift with class-aware refinement | Most expressive method in the package | Source labels are reliable and pseudo-labels stabilize | Error amplification, heavier tuning, higher variance | High | Last-stage or rescue option |

## Helper tools that are not main DA entry points

These functions are useful, but they should usually be treated as support tools
rather than primary domain adaptation algorithms:

- `kmm_weights()`: importance weighting for covariate shift.
- `label_shift_em()`: prior correction for label shift.
- `euclidean_alignment()`: trial-level alignment before feature extraction.

They are best used as part of a larger workflow, not as direct replacements for
the main methods in `domain_adaptation()`.

## A practical workflow

1. Start with your current feature matrix and benchmark `SA` and `CORAL`.
2. Compare methods using downstream predictive performance and shift metrics
   such as `distanceSummary()`.
3. If covariance structure dominates, move to `PT` and `ART`.
4. If mismatch looks nonlinear or cluster-wise, try `OT`.
5. If you have strong source labels and need class-aware refinement, try `M3D`.

## Minimal example

```r
library(DA4BCI)

res <- domain_adaptation(
  source_data,
  target_data,
  method = "coral",
  control = list(lambda = 1e-5)
)

distanceSummary(
  res$weighted_source_data,
  res$target_data,
  format = "table"
)
```

If you want a conservative comparison set, use `SA`, `CORAL`, `PT`, and `OT`
as your first benchmark panel.
