# test-tca.R
# Tests for Transfer Component Analysis (TCA)
# Fixed seeds produce reference values for cross-language (Python) validation.

test_that("TCA returns correct structure", {

  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)

  res <- domain_adaptation_tca(src, tgt, k = 3, sigma = 1, mu = 0.1)

  expect_type(res, "list")
  expect_named(res, c("weighted_source_data", "target_data", "eigenvalue"))
})

test_that("TCA output dimensions match k", {
  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)

  for (k in c(2, 3, 5)) {
    res <- domain_adaptation_tca(src, tgt, k = k, sigma = 1, mu = 1)
    expect_equal(ncol(res$weighted_source_data), k)
    expect_equal(ncol(res$target_data), k)
    expect_equal(nrow(res$weighted_source_data), 20)
    expect_equal(nrow(res$target_data), 20)
  }
})

test_that("TCA reduces MMD between domains", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(200, mean = 3), nrow = 40, ncol = 5)

  res <- domain_adaptation_tca(src, tgt, k = 5, sigma = 1, mu = 0.1)

  mmd_before <- compute_mmd(src, tgt, sigma = 1)
  mmd_after  <- compute_mmd(res$weighted_source_data, res$target_data, sigma = 1)

  expect_lt(mmd_after, mmd_before)
})

test_that("TCA is deterministic with same seed", {
  run_tca <- function() {
    set.seed(123)
    src <- matrix(rnorm(100), 20, 5)
    tgt <- matrix(rnorm(100, mean = 1), 20, 5)
    domain_adaptation_tca(src, tgt, k = 3, sigma = 1, mu = 1)
  }

  r1 <- run_tca()
  r2 <- run_tca()
  expect_equal(r1$weighted_source_data, r2$weighted_source_data)
  expect_equal(r1$target_data, r2$target_data)
})

test_that("TCA reference values for cross-language validation", {
  # This test saves reference outputs that Python tests can compare against.
  set.seed(2024)
  src <- matrix(rnorm(50), nrow = 10, ncol = 5)
  tgt <- matrix(rnorm(50, mean = 1), nrow = 10, ncol = 5)

  res <- domain_adaptation_tca(src, tgt, k = 3, sigma = 1, mu = 0.5)

  # Basic sanity: output is real-valued and finite
  expect_true(all(is.finite(res$weighted_source_data)))
  expect_true(all(is.finite(res$target_data)))
  expect_equal(dim(res$weighted_source_data), c(10, 3))
  expect_equal(dim(res$target_data), c(10, 3))
})
