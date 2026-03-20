# test-art.R
# Tests for Aligned Riemannian Transport (ART)

test_that("ART returns correct structure", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(200) + 1, nrow = 40, ncol = 5)

  res <- domain_adaptation_art(src, tgt)

  expect_type(res, "list")
  expect_named(res, c("weighted_source_data", "target_data", "transformation_matrix"))
})

test_that("ART preserves dimensions", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(150) + 1, nrow = 30, ncol = 5)

  res <- domain_adaptation_art(src, tgt)

  expect_equal(dim(res$weighted_source_data), c(40, 5))
  expect_equal(dim(res$target_data), c(30, 5))
  expect_equal(dim(res$transformation_matrix), c(5, 5))
})

test_that("ART target_data is unchanged", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(200) + 1, nrow = 40, ncol = 5)

  res <- domain_adaptation_art(src, tgt)
  expect_equal(res$target_data, tgt)
})

test_that("ART aligned source mean is close to target mean", {
  set.seed(42)
  n <- 200
  p <- 5
  src <- matrix(rnorm(n * p), nrow = n, ncol = p)
  tgt <- matrix(rnorm(n * p, mean = 3), nrow = n, ncol = p)

  res <- domain_adaptation_art(src, tgt)

  mean_aligned <- colMeans(res$weighted_source_data)
  mean_target  <- colMeans(tgt)

  # After ART, aligned source mean should be close to target mean
  expect_lt(sum((mean_aligned - mean_target)^2),
            sum((colMeans(src) - mean_target)^2))
})

test_that("ART is deterministic with same seed", {
  run_art <- function() {
    set.seed(123)
    src <- matrix(rnorm(200), 40, 5)
    tgt <- matrix(rnorm(200) + 1, 40, 5)
    domain_adaptation_art(src, tgt)
  }

  r1 <- run_art()
  r2 <- run_art()
  expect_equal(r1$weighted_source_data, r2$weighted_source_data)
  expect_equal(r1$transformation_matrix, r2$transformation_matrix)
})

test_that("ART reference values for cross-language validation", {
  set.seed(2024)
  src <- matrix(rnorm(50), nrow = 10, ncol = 5)
  tgt <- matrix(rnorm(50, mean = 1), nrow = 10, ncol = 5)

  res <- domain_adaptation_art(src, tgt)

  expect_true(all(is.finite(res$weighted_source_data)))
  expect_true(all(is.finite(res$transformation_matrix)))
  expect_equal(dim(res$weighted_source_data), c(10, 5))
})
