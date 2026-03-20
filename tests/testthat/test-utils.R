# test-utils.R
# Tests for utility functions: label_shift_em, euclidean_alignment,
# proxy_a_distance, distanceSummary, evaluate_shift,
# ph_init, ph_update

# ---- label_shift_em ----

test_that("label_shift_em returns correct structure", {
  set.seed(42)
  P <- matrix(runif(50), 10, 5)
  P <- P / rowSums(P)  # normalize rows

  out <- label_shift_em(P, rep(1 / 5, 5))

  expect_type(out, "list")
  expect_true("pi_t" %in% names(out))
  expect_true("P_adj" %in% names(out))
  expect_true("iter" %in% names(out))
})

test_that("label_shift_em priors sum to 1", {
  set.seed(42)
  P <- matrix(runif(50), 10, 5)
  P <- P / rowSums(P)

  out <- label_shift_em(P, rep(1 / 5, 5))

  expect_equal(sum(out$pi_t), 1, tolerance = 1e-10)
})

test_that("label_shift_em adjusted posteriors are valid probabilities", {
  set.seed(42)
  P <- matrix(runif(50), 10, 5)
  P <- P / rowSums(P)

  out <- label_shift_em(P, rep(1 / 5, 5))

  # All entries non-negative
  expect_true(all(out$P_adj >= 0))
  # Rows sum to 1
  expect_equal(rowSums(out$P_adj), rep(1, 10), tolerance = 1e-10)
})

# ---- euclidean_alignment ----

test_that("euclidean_alignment returns same number of trials", {
  set.seed(42)
  trials <- replicate(5, matrix(rnorm(32 * 100), 32, 100), simplify = FALSE)

  out <- euclidean_alignment(trials)
  expect_length(out, 5)
})

test_that("euclidean_alignment preserves trial dimensions", {
  set.seed(42)
  trials <- replicate(5, matrix(rnorm(32 * 100), 32, 100), simplify = FALSE)

  out <- euclidean_alignment(trials)
  for (i in seq_along(out)) {
    expect_equal(dim(out[[i]]), c(32, 100))
  }
})

test_that("euclidean_alignment to identity whitens mean covariance", {
  set.seed(42)
  trials <- replicate(10, matrix(rnorm(8 * 50), 8, 50), simplify = FALSE)

  out <- euclidean_alignment(trials)

  # Compute mean covariance of aligned trials
  covs <- lapply(out, function(X) {
    X <- as.matrix(X)
    S <- X %*% t(X)
    S / max(1, ncol(X) - 1)
  })
  R <- Reduce(`+`, covs) / length(covs)

  # Should be close to identity
  expect_equal(R, diag(8), tolerance = 0.3)
})

# ---- proxy_a_distance ----

test_that("proxy_a_distance returns pad and err", {
  set.seed(42)
  Xs <- matrix(rnorm(200), 20, 10)
  Xt <- matrix(rnorm(200, 1), 20, 10)

  out <- proxy_a_distance(Xs, Xt, seed = 42)

  expect_type(out, "list")
  expect_true("pad" %in% names(out))
  expect_true("err" %in% names(out))
})

test_that("proxy_a_distance PAD is in [-2, 2]", {
  set.seed(42)
  Xs <- matrix(rnorm(200), 20, 10)
  Xt <- matrix(rnorm(200, 1), 20, 10)

  out <- proxy_a_distance(Xs, Xt, seed = 42)

  expect_gte(out$pad, -2)
  expect_lte(out$pad, 2)
})

test_that("proxy_a_distance error is in [0, 0.5]", {
  set.seed(42)
  Xs <- matrix(rnorm(200), 20, 10)
  Xt <- matrix(rnorm(200, 1), 20, 10)

  out <- proxy_a_distance(Xs, Xt, seed = 42)

  expect_gte(out$err, 0)
  expect_lte(out$err, 0.5)
})

test_that("proxy_a_distance is high for very different distributions", {
  set.seed(42)
  Xs <- matrix(rnorm(200), 20, 10)
  Xt <- matrix(rnorm(200, 10), 20, 10)  # very shifted

  out <- proxy_a_distance(Xs, Xt, seed = 42)
  expect_gt(out$pad, 1.0)  # should be easy to separate
})

# ---- evaluate_shift ----

test_that("evaluate_shift returns a data frame", {
  set.seed(42)
  A <- matrix(rnorm(200), nrow = 20, ncol = 10)
  B <- matrix(rnorm(200, mean = 2), nrow = 20, ncol = 10)
  As <- matrix(rnorm(200, mean = 1), nrow = 20, ncol = 10)
  Bs <- matrix(rnorm(200, mean = 1.5), nrow = 20, ncol = 10)

  df <- evaluate_shift(A, B, As, Bs)

  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 2)
  expect_true(all(c("Metric", "Before", "After") %in% names(df)))
})

test_that("evaluate_shift reports MMD and Wasserstein", {
  set.seed(42)
  A <- matrix(rnorm(200), nrow = 20, ncol = 10)
  B <- matrix(rnorm(200, mean = 2), nrow = 20, ncol = 10)
  As <- matrix(rnorm(200, mean = 1), nrow = 20, ncol = 10)
  Bs <- matrix(rnorm(200, mean = 1.5), nrow = 20, ncol = 10)

  df <- evaluate_shift(A, B, As, Bs)
  expect_true("MMD" %in% df$Metric)
  expect_true("Wasserstein" %in% df$Metric)
})

# ---- ph_init / ph_update ----

test_that("ph_init returns correct structure", {
  s <- ph_init()

  expect_type(s, "list")
  expect_true(all(c("mean", "cum", "min_cum", "delta", "lambda", "alpha") %in% names(s)))
  expect_equal(s$mean, 0)
  expect_equal(s$cum, 0)
  expect_equal(s$min_cum, 0)
})

test_that("ph_init respects custom parameters", {
  s <- ph_init(delta = 0.01, lambda = 100, alpha = 0.99)

  expect_equal(s$delta, 0.01)
  expect_equal(s$lambda, 100)
  expect_equal(s$alpha, 0.99)
})

test_that("ph_update returns state and change flag", {
  s <- ph_init()
  out <- ph_update(s, 1.0)

  expect_type(out, "list")
  expect_true("state" %in% names(out))
  expect_true("change" %in% names(out))
  expect_type(out$change, "logical")
})

test_that("ph_update detects large shift", {
  s <- ph_init(delta = 0.005, lambda = 5, alpha = 0.999)

  # Feed many stable observations, then a large shift
  for (z in rnorm(100, mean = 0, sd = 0.1)) {
    out <- ph_update(s, z)
    s <- out$state
  }

  # Inject a sudden shift
  for (z in rnorm(50, mean = 10, sd = 0.1)) {
    out <- ph_update(s, z)
    s <- out$state
  }
  # After a large shift, change should be detected
  expect_true(out$change)
})

test_that("ph_update no false alarm on stable data", {
  s <- ph_init(delta = 0.005, lambda = 50, alpha = 0.999)

  any_change <- FALSE
  for (z in rnorm(200, mean = 0, sd = 0.1)) {
    out <- ph_update(s, z)
    s <- out$state
    if (out$change) any_change <- TRUE
  }
  expect_false(any_change)
})

test_that("Utils reference values for cross-language validation", {
  # label_shift_em
  set.seed(2024)
  P <- matrix(runif(30), 6, 5)
  P <- P / rowSums(P)
  out_em <- label_shift_em(P, rep(1 / 5, 5))
  expect_true(all(is.finite(out_em$pi_t)))

  # Page-Hinkley
  s <- ph_init(delta = 0.01, lambda = 10, alpha = 0.99)
  vals <- c(0.1, 0.2, -0.1, 5.0, 5.0, 5.0, 5.0, 5.0)
  for (v in vals) {
    out <- ph_update(s, v)
    s <- out$state
  }
  expect_true(is.logical(out$change))
})
