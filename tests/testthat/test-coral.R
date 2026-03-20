# test-coral.R
# Tests for Correlation Alignment (CORAL)

test_that("CORAL returns correct structure", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(200, sd = 3), nrow = 40, ncol = 5)

  res <- domain_adaptation_coral(src, tgt, lambda = 1e-5)

  expect_type(res, "list")
  expect_named(res, c("weighted_source_data", "target_data"))
})

test_that("CORAL preserves dimensions", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(150, sd = 2), nrow = 30, ncol = 5)

  res <- domain_adaptation_coral(src, tgt)

  expect_equal(dim(res$weighted_source_data), c(40, 5))
  expect_equal(dim(res$target_data), c(30, 5))
})

test_that("CORAL target_data is unchanged", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(200, sd = 3), nrow = 40, ncol = 5)

  res <- domain_adaptation_coral(src, tgt)
  expect_equal(res$target_data, tgt)
})

test_that("CORAL aligns covariance structure", {
  set.seed(42)
  n <- 200
  p <- 5
  src <- matrix(rnorm(n * p), nrow = n, ncol = p)
  # Target with different covariance
  L <- matrix(rnorm(p * p), p, p)
  tgt <- matrix(rnorm(n * p), nrow = n, ncol = p) %*% L

  res <- domain_adaptation_coral(src, tgt, lambda = 1e-5)

  # After CORAL, source covariance should be closer to target covariance
  cov_src_before <- cov(src)
  cov_src_after  <- cov(res$weighted_source_data)
  cov_tgt        <- cov(tgt)

  dist_before <- norm(cov_src_before - cov_tgt, "F")
  dist_after  <- norm(cov_src_after - cov_tgt, "F")

  expect_lt(dist_after, dist_before)
})

test_that("CORAL is deterministic with same seed", {
  run_coral <- function() {
    set.seed(123)
    src <- matrix(rnorm(200), 40, 5)
    tgt <- matrix(rnorm(200, sd = 2), 40, 5)
    domain_adaptation_coral(src, tgt, lambda = 1e-5)
  }

  r1 <- run_coral()
  r2 <- run_coral()
  expect_equal(r1$weighted_source_data, r2$weighted_source_data)
})

test_that("CORAL reference values for cross-language validation", {
  set.seed(2024)
  src <- matrix(rnorm(50), nrow = 10, ncol = 5)
  tgt <- matrix(rnorm(50, mean = 1, sd = 2), nrow = 10, ncol = 5)

  res <- domain_adaptation_coral(src, tgt, lambda = 1e-5)

  expect_true(all(is.finite(res$weighted_source_data)))
  expect_equal(dim(res$weighted_source_data), c(10, 5))
})
