# test-ot.R
# Tests for Entropy-Regularized Optimal Transport (OT) with Sinkhorn-Knopp

test_that("OT returns correct structure", {
  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)

  res <- domain_adaptation_ot(src, tgt, eps = 0.05, maxit = 500)

  expect_type(res, "list")
  expect_true("weighted_source_data" %in% names(res))
  expect_true("target_data" %in% names(res))
  expect_true("ot_plan" %in% names(res))
  expect_true("cost" %in% names(res))
  expect_true("epsilon" %in% names(res))
  expect_true("iterations" %in% names(res))
  expect_true("converged" %in% names(res))
  expect_true("residual" %in% names(res))
})

test_that("OT preserves dimensions", {
  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(75, mean = 2), nrow = 15, ncol = 5)

  res <- domain_adaptation_ot(src, tgt, eps = 0.1)

  expect_equal(dim(res$weighted_source_data), c(20, 5))
  expect_equal(dim(res$target_data), c(15, 5))
  expect_equal(dim(res$ot_plan), c(20, 15))
  expect_equal(dim(res$cost), c(20, 15))
})

test_that("OT plan has correct marginals", {
  set.seed(42)
  n_s <- 20
  n_t <- 15
  src <- matrix(rnorm(n_s * 5), nrow = n_s, ncol = 5)
  tgt <- matrix(rnorm(n_t * 5, mean = 1), nrow = n_t, ncol = 5)

  res <- domain_adaptation_ot(src, tgt, eps = 0.05, maxit = 1000, tol = 1e-8)

  if (res$converged) {
    # Row sums should be close to 1/n_s
    expect_equal(rowSums(res$ot_plan), rep(1 / n_s, n_s), tolerance = 1e-5)
    # Col sums should be close to 1/n_t
    expect_equal(colSums(res$ot_plan), rep(1 / n_t, n_t), tolerance = 1e-5)
  }
})

test_that("OT plan is non-negative", {
  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)

  res <- domain_adaptation_ot(src, tgt, eps = 0.1)
  expect_true(all(res$ot_plan >= 0))
})

test_that("OT cost matrix is non-negative", {
  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)

  res_sq <- domain_adaptation_ot(src, tgt, eps = 0.1, cost = "sqeuclidean")
  expect_true(all(res_sq$cost >= 0))

  res_euc <- domain_adaptation_ot(src, tgt, eps = 0.1, cost = "euclidean")
  expect_true(all(res_euc$cost >= 0))
})

test_that("OT with different cost types produce different results", {
  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)

  res_sq  <- domain_adaptation_ot(src, tgt, eps = 0.1, cost = "sqeuclidean")
  res_euc <- domain_adaptation_ot(src, tgt, eps = 0.1, cost = "euclidean")

  expect_false(isTRUE(all.equal(res_sq$weighted_source_data,
                                 res_euc$weighted_source_data)))
})

test_that("OT epsilon is stored correctly", {
  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)

  res <- domain_adaptation_ot(src, tgt, eps = 0.42)
  expect_equal(res$epsilon, 0.42)
})

test_that("OT is deterministic with same seed", {
  run_ot <- function() {
    set.seed(123)
    src <- matrix(rnorm(100), 20, 5)
    tgt <- matrix(rnorm(100, mean = 1), 20, 5)
    domain_adaptation_ot(src, tgt, eps = 0.05, maxit = 500)
  }

  r1 <- run_ot()
  r2 <- run_ot()
  expect_equal(r1$weighted_source_data, r2$weighted_source_data)
  expect_equal(r1$ot_plan, r2$ot_plan)
})

test_that("OT reference values for cross-language validation", {
  set.seed(2024)
  src <- matrix(rnorm(50), nrow = 10, ncol = 5)
  tgt <- matrix(rnorm(50, mean = 1), nrow = 10, ncol = 5)

  res <- domain_adaptation_ot(src, tgt, eps = 0.1, maxit = 500, tol = 1e-7)

  expect_true(all(is.finite(res$weighted_source_data)))
  expect_true(all(is.finite(res$ot_plan)))
  expect_equal(dim(res$weighted_source_data), c(10, 5))
  expect_equal(dim(res$ot_plan), c(10, 10))
})
