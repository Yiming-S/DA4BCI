# DA4BCI 0.1.0

## Fixed (correctness — ported from DA4BCI-Python, verified by a cross-language golden test)

> These change the numerical output of GFK, CORAL, the Grassmann geodesic
> distance, and any RBF-based metric / KMM weight. Re-run any analysis that used
> the previous versions.

* **GFK was not a valid geodesic flow.** `domain_adaptation_gfk` built the kernel
  from `t(U0) %*% U1` on the *full* orthonormal bases, which is always orthogonal,
  so the principal angles were structurally trivial and `G` was effectively the
  identity (a no-op) or numerical noise. It now uses the principal angles between
  the source/target PCA *subspaces* with the complete geodesic-flow integral
  (lambda1/lambda2/lambda3 terms), clamps the subspace dimension to each domain's
  usable rank, and projects features by `G^(1/2)`.
* **CORAL whitened the source with the wrong Cholesky factor.** It used R's upper
  factor `chol(solve(cov_source))` directly, so the source was not whitened and
  the aligned covariance did not match the target. It now uses the lower factor
  `t(chol(...))`; the aligned covariance matches the target to the regularization
  floor.
* **`compute_geodesic` crashed for unequal sample sizes.** It built the subspace
  basis in *sample* space (`qr.Q`, shape `n x d`), requiring `n_s == n_t`. It now
  uses the feature-space right singular vectors (shape `p x d`), well-defined for
  any sample sizes, with the default subspace dimension taken from the *centered*
  rank.
* **`sigma_med` is now reproducible by default** (`seed = 0` instead of `NULL`),
  so the RBF bandwidth — and every MMD / Energy / KMM value built on it — no
  longer changes run-to-run when the inputs are subsampled.

Initial release.

* Unified interface to ten domain-adaptation methods for aligning EEG feature
  distributions across BCI sessions and subjects: subspace alignment (`domain_adaptation_sa`),
  CORAL (`domain_adaptation_coral`), geodesic flow kernel (`domain_adaptation_gfk`),
  transfer component analysis (`domain_adaptation_tca`), maximum independence DA
  (`domain_adaptation_mida`), Riemannian rotation (`domain_adaptation_riemannian`),
  parallel transport (`domain_adaptation_pt`), aligned Riemannian transport
  (`domain_adaptation_art`), optimal transport (`domain_adaptation_ot`), and the
  class-aware multi-stage method (`domain_adaptation_m3d`).
* Distribution-shift metrics: `compute_mmd`, `compute_energy`, `compute_wasserstein`,
  `compute_mahalanobis`, plus `distanceSummary`, `evaluate_shift`, and `proxy_a_distance`.
* Supporting utilities: `euclidean_alignment`, `kmm_weights`, `label_shift_em`,
  Page-Hinkley drift detection (`ph_init`/`ph_update`), and before/after PCA/t-SNE
  visualization (`plot_data_comparison`).
* Algorithm-selection guidance in `ALGORITHM_SELECTION_GUIDE.md`.
