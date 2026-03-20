# test-domain_adaptation.R
# Tests for the unified domain_adaptation() interface

test_that("domain_adaptation dispatches all methods without error", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(200, mean = 2), nrow = 40, ncol = 5)

  methods_no_labels <- c("tca", "sa", "mida", "rd", "coral", "gfk", "art", "pt", "ot")

  for (m in methods_no_labels) {
    res <- domain_adaptation(src, tgt, method = m)
    expect_type(res, "list")
    expect_true("weighted_source_data" %in% names(res))
  }
})

test_that("domain_adaptation m3d requires source_labels", {
  set.seed(42)
  src <- matrix(rnorm(200), nrow = 40, ncol = 5)
  tgt <- matrix(rnorm(200, mean = 2), nrow = 40, ncol = 5)

  expect_error(domain_adaptation(src, tgt, method = "m3d"))
})

test_that("domain_adaptation m3d works with labels", {
  set.seed(42)
  src <- matrix(rnorm(80 * 8), nrow = 80, ncol = 8)
  tgt <- matrix(rnorm(80 * 8, mean = 1), nrow = 80, ncol = 8)
  labels <- sample(1:2, 80, replace = TRUE)

  res <- domain_adaptation(src, tgt, method = "m3d",
                            control = list(source_labels = labels,
                                           l_iter = 2, max_dim = 6))
  expect_type(res, "list")
  expect_true("weighted_source_data" %in% names(res))
})

test_that("domain_adaptation rejects invalid method", {
  src <- matrix(rnorm(20), 4, 5)
  tgt <- matrix(rnorm(20), 4, 5)

  expect_error(domain_adaptation(src, tgt, method = "invalid_method"))
})

test_that("domain_adaptation default method is sa", {
  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)

  res_default <- domain_adaptation(src, tgt)
  res_sa      <- domain_adaptation(src, tgt, method = "sa")

  expect_equal(res_default$weighted_source_data, res_sa$weighted_source_data)
})

test_that("domain_adaptation control parameters are passed through", {
  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)

  # TCA with different k values should give different dimensions
  res_k3 <- domain_adaptation(src, tgt, method = "tca",
                               control = list(k = 3, sigma = 1, mu = 0.1))
  res_k5 <- domain_adaptation(src, tgt, method = "tca",
                               control = list(k = 5, sigma = 1, mu = 0.1))

  expect_equal(ncol(res_k3$weighted_source_data), 3)
  expect_equal(ncol(res_k5$weighted_source_data), 5)
})

test_that("domain_adaptation OT control parameters pass through", {
  set.seed(42)
  src <- matrix(rnorm(100), nrow = 20, ncol = 5)
  tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)

  res <- domain_adaptation(src, tgt, method = "ot",
                            control = list(eps = 0.2, maxit = 100))
  expect_equal(res$epsilon, 0.2)
})

test_that("Unified interface reference for cross-language validation", {
  set.seed(2024)
  src <- matrix(rnorm(50), nrow = 10, ncol = 5)
  tgt <- matrix(rnorm(50, mean = 1), nrow = 10, ncol = 5)

  # Run each method through the unified interface
  for (m in c("sa", "coral", "rd")) {
    res <- domain_adaptation(src, tgt, method = m)
    expect_true(all(is.finite(res$weighted_source_data)), label = m)
  }
})
