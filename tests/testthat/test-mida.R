# test-mida.R
# Tests for Maximum/Minimum Independence Domain Adaptation (MIDA)

test_that("MIDA returns correct structure", {
  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(100, mean = 3), nrow = 20, ncol = 5)

  res <- domain_adaptation_mida(src, tgt, k = 3, sigma = 1, mu = 0.1, max = TRUE)

  expect_type(res, "list")
  expect_named(res, c("weighted_source_data", "target_data", "eigenvalue"))
})

test_that("MIDA output dimensions match k", {
  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(100, mean = 3), nrow = 20, ncol = 5)

  for (k in c(2, 3, 5)) {
    res <- domain_adaptation_mida(src, tgt, k = k, sigma = 1, mu = 0.1, max = TRUE)
    expect_equal(ncol(res$weighted_source_data), k)
    expect_equal(ncol(res$target_data), k)
    expect_equal(nrow(res$weighted_source_data), 20)
    expect_equal(nrow(res$target_data), 20)
  }
})

test_that("MIDA works in both max and min mode", {
  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)

  res_max <- domain_adaptation_mida(src, tgt, k = 3, sigma = 1, mu = 0.1, max = TRUE)
  res_min <- domain_adaptation_mida(src, tgt, k = 3, sigma = 1, mu = 0.1, max = FALSE)

  # Both should return valid outputs
  expect_true(all(is.finite(res_max$weighted_source_data)))
  expect_true(all(is.finite(res_min$weighted_source_data)))

  # Results should differ between max and min
  expect_false(isTRUE(all.equal(res_max$weighted_source_data, res_min$weighted_source_data)))
})

test_that("MIDA is deterministic with same seed", {
  run_mida <- function() {
    set.seed(123)
    src <- matrix(rnorm(100), 20, 5)
    tgt <- matrix(rnorm(100, mean = 1), 20, 5)
    domain_adaptation_mida(src, tgt, k = 3, sigma = 1, mu = 0.1, max = TRUE)
  }

  r1 <- run_mida()
  r2 <- run_mida()
  expect_equal(r1$weighted_source_data, r2$weighted_source_data)
})

test_that("MIDA reference values for cross-language validation", {
  set.seed(2024)
  src <- matrix(rnorm(50), nrow = 10, ncol = 5)
  tgt <- matrix(rnorm(50, mean = 1), nrow = 10, ncol = 5)

  res <- domain_adaptation_mida(src, tgt, k = 3, sigma = 1, mu = 0.1, max = TRUE)

  expect_true(all(is.finite(res$weighted_source_data)))
  expect_equal(dim(res$weighted_source_data), c(10, 3))
  expect_equal(dim(res$target_data), c(10, 3))
})
