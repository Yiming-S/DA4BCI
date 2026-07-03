# Regression guards for the correctness fixes ported from DA4BCI-Python.
# Each test would FAIL against the pre-fix behaviour.

test_that("GFK projects by G^(1/2) and yields a non-trivial kernel", {
  set.seed(7)
  src <- matrix(rnorm(60 * 8), 60, 8)
  # Give the target a genuinely different principal subspace via a rotation.
  R <- qr.Q(qr(matrix(rnorm(64), 8, 8)))
  tgt <- matrix(rnorm(60 * 8), 60, 8) %*% R
  res <- domain_adaptation_gfk(src, tgt, dim_subspace = 4)

  eg <- eigen((res$G + t(res$G)) / 2, symmetric = TRUE)
  G_half <- eg$vectors %*% diag(sqrt(pmax(eg$values, 0))) %*% t(eg$vectors)

  # Features are mapped by G^(1/2), not by G (the old behaviour).
  expect_equal(res$weighted_source_data, src %*% G_half, tolerance = 1e-8)
  expect_false(isTRUE(all.equal(res$weighted_source_data, src %*% res$G,
                                tolerance = 1e-6)))
  # Different subspaces -> the flow kernel is not a scaled identity (the old
  # full-basis construction gave trivial angles and a ~identity/no-op G).
  expect_gt(max(eg$values) - min(eg$values), 0.1)
  expect_gt(max(abs(res$G[upper.tri(res$G)])), 1e-6)
})

test_that("CORAL aligns source covariance to the target (not off by ~15x)", {
  set.seed(11)
  p <- 5
  src <- matrix(rnorm(500 * p), 500, p)
  L <- matrix(rnorm(p * p), p, p)
  tgt <- matrix(rnorm(500 * p), 500, p) %*% L
  res <- domain_adaptation_coral(src, tgt, lambda = 1e-5)

  # Aligned-source covariance matches target to the regularization floor; the
  # old upper-factor whitening left it off by more than an order of magnitude.
  expect_lt(norm(cov(res$weighted_source_data) - cov(tgt), "F"),
            0.1 * norm(cov(tgt), "F"))
})

test_that("compute_geodesic works when n_s != n_t (no sample-space crash)", {
  set.seed(13)
  src <- matrix(rnorm(50 * 8), 50, 8)
  tgt <- matrix(rnorm(40 * 8), 40, 8)          # different sample count
  d <- NULL
  expect_silent(d <- compute_geodesic(src, tgt))
  expect_true(is.finite(d))
  expect_gte(d, 0)
})

test_that("sigma_med is reproducible by default even when it subsamples", {
  set.seed(1)
  X <- matrix(rnorm(300 * 4), 300, 4)
  Y <- matrix(rnorm(300 * 4), 300, 4)          # N = 600 > m = 400 -> subsamples
  expect_equal(sigma_med(X, Y), sigma_med(X, Y))
  expect_true(is.finite(sigma_med(X, Y)) && sigma_med(X, Y) > 0)
})
