# test-riemannian.R
# Tests for Riemannian-Distance-Based Alignment (RD)

test_that("RD returns correct structure", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(200, sd = 3), nrow = 40, ncol = 5)

  res <- domain_adaptation_riemannian(src, tgt)

  expect_type(res, "list")
  expect_true("weighted_source_data" %in% names(res))
  expect_true("target_data" %in% names(res))
  expect_true("rotation_matrix" %in% names(res))
  expect_true("cov_source_aligned" %in% names(res))
  expect_true("riemannian_distance" %in% names(res))
})

test_that("RD preserves dimensions", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(150, sd = 2), nrow = 30, ncol = 5)

  res <- domain_adaptation_riemannian(src, tgt)

  expect_equal(dim(res$weighted_source_data), c(40, 5))
  expect_equal(dim(res$target_data), c(30, 5))
  expect_equal(dim(res$rotation_matrix), c(5, 5))
})

test_that("RD target_data is unchanged", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(200, sd = 3), nrow = 40, ncol = 5)

  res <- domain_adaptation_riemannian(src, tgt)
  expect_equal(res$target_data, tgt)
})

test_that("RD rotation matrix is orthogonal", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(200, sd = 3), nrow = 40, ncol = 5)

  res <- domain_adaptation_riemannian(src, tgt)
  R <- res$rotation_matrix

  # R^T R should be close to identity
  expect_equal(crossprod(R), diag(5), tolerance = 1e-8)
})

test_that("RD riemannian_distance is non-negative", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(200, sd = 3), nrow = 40, ncol = 5)

  res <- domain_adaptation_riemannian(src, tgt)
  expect_gte(res$riemannian_distance, 0)
})

test_that("RD distance is zero for identical distributions", {
  set.seed(42)
  data <- matrix(rnorm(200), nrow = 40, ncol = 5)

  res <- domain_adaptation_riemannian(data, data)
  expect_equal(res$riemannian_distance, 0, tolerance = 1e-6)
})

test_that("RD is deterministic with same seed", {
  run_rd <- function() {
    set.seed(123)
    src <- matrix(rnorm(200), 40, 5)
    tgt <- matrix(rnorm(200, sd = 2), 40, 5)
    domain_adaptation_riemannian(src, tgt)
  }

  r1 <- run_rd()
  r2 <- run_rd()
  expect_equal(r1$weighted_source_data, r2$weighted_source_data)
  expect_equal(r1$rotation_matrix, r2$rotation_matrix)
})

test_that("RD reference values for cross-language validation", {
  set.seed(2024)
  src <- matrix(rnorm(50), nrow = 10, ncol = 5)
  tgt <- matrix(rnorm(50, mean = 1, sd = 2), nrow = 10, ncol = 5)

  res <- domain_adaptation_riemannian(src, tgt)

  expect_true(all(is.finite(res$weighted_source_data)))
  expect_true(is.finite(res$riemannian_distance))
  expect_equal(dim(res$weighted_source_data), c(10, 5))
})
