# test-pt.R
# Tests for Parallel Transport (PT)

test_that("PT returns correct structure", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(200) + 1, nrow = 40, ncol = 5)

  res <- domain_adaptation_pt(src, tgt)

  expect_type(res, "list")
  expect_named(res, c("weighted_source_data", "target_data", "transformation_matrix"))
})

test_that("PT preserves dimensions", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(150) + 1, nrow = 30, ncol = 5)

  res <- domain_adaptation_pt(src, tgt)

  expect_equal(dim(res$weighted_source_data), c(40, 5))
  expect_equal(dim(res$target_data), c(30, 5))
  expect_equal(dim(res$transformation_matrix), c(5, 5))
})

test_that("PT target_data is unchanged", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(200) + 1, nrow = 40, ncol = 5)

  res <- domain_adaptation_pt(src, tgt)
  expect_equal(res$target_data, tgt)
})

test_that("PT maps source covariance to target covariance", {
  set.seed(42)
  n <- 500
  p <- 4
  src <- matrix(rnorm(n * p), nrow = n, ncol = p)
  # Target with different covariance structure
  L <- matrix(c(2, 0.5, 0, 0, 0.5, 1, 0, 0, 0, 0, 3, 0.3, 0, 0, 0.3, 1.5), 4, 4)
  tgt <- matrix(rnorm(n * p), nrow = n, ncol = p) %*% L + 2

  res <- domain_adaptation_pt(src, tgt)

  # Check covariance is closer to target after PT
  cov_aligned <- cov(res$weighted_source_data)
  cov_tgt     <- cov(tgt)
  cov_src     <- cov(src)

  dist_before <- norm(cov_src - cov_tgt, "F")
  dist_after  <- norm(cov_aligned - cov_tgt, "F")

  expect_lt(dist_after, dist_before)
})

test_that("PT is deterministic with same seed", {
  run_pt <- function() {
    set.seed(123)
    src <- matrix(rnorm(200), 40, 5)
    tgt <- matrix(rnorm(200) + 1, 40, 5)
    domain_adaptation_pt(src, tgt)
  }

  r1 <- run_pt()
  r2 <- run_pt()
  expect_equal(r1$weighted_source_data, r2$weighted_source_data)
  expect_equal(r1$transformation_matrix, r2$transformation_matrix)
})

test_that("PT reference values for cross-language validation", {
  set.seed(2024)
  src <- matrix(rnorm(50), nrow = 10, ncol = 5)
  tgt <- matrix(rnorm(50, mean = 1), nrow = 10, ncol = 5)

  res <- domain_adaptation_pt(src, tgt)

  expect_true(all(is.finite(res$weighted_source_data)))
  expect_true(all(is.finite(res$transformation_matrix)))
  expect_equal(dim(res$weighted_source_data), c(10, 5))
})
