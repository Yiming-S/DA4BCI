# test-sa.R
# Tests for Subspace Alignment (SA)

test_that("SA returns correct structure", {
  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)

  res <- domain_adaptation_sa(src, tgt, k = 3)

  expect_type(res, "list")
  expect_named(res, c("weighted_source_data", "target_data", "eigenvalue"))
})

test_that("SA output dimensions match k", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(200, mean = 2), nrow = 40, ncol = 5)

  for (k in c(2, 3, 5)) {
    res <- domain_adaptation_sa(src, tgt, k = k)
    expect_equal(ncol(res$weighted_source_data), k)
    expect_equal(ncol(res$target_data), k)
    expect_equal(nrow(res$weighted_source_data), 40)
    expect_equal(nrow(res$target_data), 40)
  }
})

test_that("SA k is capped at number of features", {
  set.seed(42)
  src <- matrix(rnorm(60), nrow = 20, ncol = 3)
  tgt <- matrix(rnorm(60, mean = 1), nrow = 20, ncol = 3)

  # Request k=10 but only 3 features
  res <- domain_adaptation_sa(src, tgt, k = 10)
  expect_equal(ncol(res$weighted_source_data), 3)
})

test_that("SA alignment matrix W is k x k", {
  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)

  res <- domain_adaptation_sa(src, tgt, k = 3)
  expect_equal(dim(res$eigenvalue), c(3, 3))
})

test_that("SA is deterministic with same seed", {
  run_sa <- function() {
    set.seed(123)
    src <- matrix(rnorm(100), 20, 5)
    tgt <- matrix(rnorm(100, mean = 1), 20, 5)
    domain_adaptation_sa(src, tgt, k = 3)
  }

  r1 <- run_sa()
  r2 <- run_sa()
  expect_equal(r1$weighted_source_data, r2$weighted_source_data)
})

test_that("SA reference values for cross-language validation", {
  set.seed(2024)
  src <- matrix(rnorm(50), nrow = 10, ncol = 5)
  tgt <- matrix(rnorm(50, mean = 1), nrow = 10, ncol = 5)

  res <- domain_adaptation_sa(src, tgt, k = 3)

  expect_true(all(is.finite(res$weighted_source_data)))
  expect_true(all(is.finite(res$target_data)))
  expect_equal(dim(res$weighted_source_data), c(10, 3))
  expect_equal(dim(res$target_data), c(10, 3))
})
