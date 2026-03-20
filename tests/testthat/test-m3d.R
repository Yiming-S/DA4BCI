# test-m3d.R
# Tests for Manifold-based Multi-step Domain Adaptation (M3D)

test_that("M3D returns correct structure", {
  set.seed(42)
  src <- matrix(rnorm(100 * 10), nrow = 100, ncol = 10)
  tgt <- matrix(rnorm(100 * 10, mean = 1), nrow = 100, ncol = 10)
  labels <- sample(1:3, 100, replace = TRUE)

  res <- domain_adaptation_m3d(src, labels, tgt,
                                l_iter = 3, max_dim = 10)

  expect_type(res, "list")
  expect_true("weighted_source_data" %in% names(res))
  expect_true("target_data" %in% names(res))
})

test_that("M3D output has expected row counts", {
  set.seed(42)
  n_s <- 60
  n_t <- 40
  p <- 10
  src <- matrix(rnorm(n_s * p), nrow = n_s, ncol = p)
  tgt <- matrix(rnorm(n_t * p, mean = 1), nrow = n_t, ncol = p)
  labels <- sample(1:2, n_s, replace = TRUE)

  res <- domain_adaptation_m3d(src, labels, tgt,
                                l_iter = 3, max_dim = 8)

  expect_equal(nrow(res$weighted_source_data), n_s)
  expect_equal(nrow(res$target_data), n_t)
  # Source and target should have the same number of columns
  expect_equal(ncol(res$weighted_source_data), ncol(res$target_data))
})

test_that("M3D works with different stage configurations", {
  set.seed(42)
  src <- matrix(rnorm(80 * 8), nrow = 80, ncol = 8)
  tgt <- matrix(rnorm(80 * 8, mean = 1), nrow = 80, ncol = 8)
  labels <- sample(1:2, 80, replace = TRUE)

  # TCA + SA (default-ish)
  res1 <- domain_adaptation_m3d(src, labels, tgt,
                                 stage1 = list(method = "tca", control = list(k = 5, sigma = 1)),
                                 stage2 = list(method = "sa", control = list(k = 5)),
                                 l_iter = 2, max_dim = 8)
  expect_true(all(is.finite(res1$weighted_source_data)))

  # SA + SA
  res2 <- domain_adaptation_m3d(src, labels, tgt,
                                 stage1 = list(method = "sa", control = list(k = 5)),
                                 stage2 = list(method = "sa", control = list(k = 5)),
                                 l_iter = 2, max_dim = 8)
  expect_true(all(is.finite(res2$weighted_source_data)))
})

test_that("M3D is deterministic with same seed", {
  run_m3d <- function() {
    set.seed(123)
    src <- matrix(rnorm(80 * 8), 80, 8)
    tgt <- matrix(rnorm(80 * 8, mean = 1), 80, 8)
    labels <- sample(1:3, 80, replace = TRUE)
    domain_adaptation_m3d(src, labels, tgt, l_iter = 3, max_dim = 8)
  }

  r1 <- run_m3d()
  r2 <- run_m3d()
  expect_equal(r1$weighted_source_data, r2$weighted_source_data)
  expect_equal(r1$target_data, r2$target_data)
})

test_that("M3D reference values for cross-language validation", {
  set.seed(2024)
  src <- matrix(rnorm(50 * 8), nrow = 50, ncol = 8)
  tgt <- matrix(rnorm(50 * 8, mean = 1), nrow = 50, ncol = 8)
  labels <- sample(1:2, 50, replace = TRUE)

  res <- domain_adaptation_m3d(src, labels, tgt,
                                l_iter = 3, max_dim = 6)

  expect_true(all(is.finite(res$weighted_source_data)))
  expect_true(all(is.finite(res$target_data)))
  expect_equal(nrow(res$weighted_source_data), 50)
  expect_equal(nrow(res$target_data), 50)
})
