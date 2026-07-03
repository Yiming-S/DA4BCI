# DA4BCI 0.1.0

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
