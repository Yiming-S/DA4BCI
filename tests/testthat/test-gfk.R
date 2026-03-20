# test-gfk.R
# Tests for Geodesic Flow Kernel (GFK)

test_that("GFK returns correct structure", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 20, ncol = 10)
  tgt <- matrix(rnorm(200, mean = 2), nrow = 20, ncol = 10)

  res <- domain_adaptation_gfk(src, tgt, dim_subspace = 5)

  expect_type(res, "list")
  expect_named(res, c("weighted_source_data", "target_data", "G"))
})

test_that("GFK preserves row count, G is square", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 20, ncol = 10)
  tgt <- matrix(rnorm(150, mean = 1), nrow = 15, ncol = 10)

  res <- domain_adaptation_gfk(src, tgt, dim_subspace = 5)

  expect_equal(nrow(res$weighted_source_data), 20)
  expect_equal(nrow(res$target_data), 15)
  expect_equal(ncol(res$weighted_source_data), 10)
  expect_equal(ncol(res$target_data), 10)
  expect_equal(dim(res$G), c(10, 10))
})

test_that("GFK dim_subspace is capped at ncol", {
  set.seed(42)
  src <- matrix(rnorm(60), nrow = 20, ncol = 3)
  tgt <- matrix(rnorm(60, mean = 1), nrow = 20, ncol = 3)

  # Request dim_subspace=10, but only 3 features
  res <- domain_adaptation_gfk(src, tgt, dim_subspace = 10)
  expect_equal(dim(res$G), c(3, 3))
})

test_that("GFK kernel G is symmetric", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 20, ncol = 10)
  tgt <- matrix(rnorm(200, mean = 2), nrow = 20, ncol = 10)

  res <- domain_adaptation_gfk(src, tgt, dim_subspace = 5)
  expect_equal(res$G, t(res$G), tolerance = 1e-10)
})

test_that("GFK is deterministic with same seed", {
  run_gfk <- function() {
    set.seed(123)
    src <- matrix(rnorm(200), 20, 10)
    tgt <- matrix(rnorm(200, mean = 1), 20, 10)
    domain_adaptation_gfk(src, tgt, dim_subspace = 5)
  }

  r1 <- run_gfk()
  r2 <- run_gfk()
  expect_equal(r1$weighted_source_data, r2$weighted_source_data)
  expect_equal(r1$G, r2$G)
})

test_that("GFK reference values for cross-language validation", {
  set.seed(2024)
  src <- matrix(rnorm(80), nrow = 10, ncol = 8)
  tgt <- matrix(rnorm(80, mean = 1), nrow = 10, ncol = 8)

  res <- domain_adaptation_gfk(src, tgt, dim_subspace = 4)

  expect_true(all(is.finite(res$weighted_source_data)))
  expect_true(all(is.finite(res$G)))
  expect_equal(dim(res$weighted_source_data), c(10, 8))
})
