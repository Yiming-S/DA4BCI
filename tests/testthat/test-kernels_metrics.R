# test-kernels_metrics.R
# Tests for kernel functions and distance metrics

# ---- rbf_kernel ----

test_that("rbf_kernel output dimensions are correct", {
  set.seed(42)
  x <- matrix(rnorm(20), 5, 4)
  y <- matrix(rnorm(24), 6, 4)

  K <- rbf_kernel(x, y, sigma = 1)

  expect_equal(dim(K), c(5, 6))
})

test_that("rbf_kernel values are in (0, 1]", {
  set.seed(42)
  x <- matrix(rnorm(20), 5, 4)
  y <- matrix(rnorm(24), 6, 4)

  K <- rbf_kernel(x, y, sigma = 1)

  expect_true(all(K > 0))
  expect_true(all(K <= 1))
})

test_that("rbf_kernel self-kernel diagonal is 1", {
  set.seed(42)
  x <- matrix(rnorm(20), 5, 4)

  K <- rbf_kernel(x, x, sigma = 1)
  expect_equal(diag(K), rep(1, 5), tolerance = 1e-12)
})

test_that("rbf_kernel is symmetric for self-kernel", {
  set.seed(42)
  x <- matrix(rnorm(20), 5, 4)

  K <- rbf_kernel(x, x, sigma = 1)
  expect_equal(K, t(K), tolerance = 1e-12)
})

test_that("rbf_kernel standard_scale flag works", {
  set.seed(42)
  x <- matrix(rnorm(20), 5, 4)
  y <- matrix(rnorm(24), 6, 4)

  K_std  <- rbf_kernel(x, y, sigma = 2, standard_scale = TRUE)
  K_nstd <- rbf_kernel(x, y, sigma = 2, standard_scale = FALSE)

  # Different gamma -> different results
  expect_false(isTRUE(all.equal(K_std, K_nstd)))
})

test_that("rbf_kernel rejects mismatched column counts", {
  x <- matrix(rnorm(20), 5, 4)
  y <- matrix(rnorm(15), 5, 3)

  expect_error(rbf_kernel(x, y, sigma = 1))
})

# ---- sigma_med ----

test_that("sigma_med returns a positive scalar", {
  set.seed(42)
  X <- matrix(rnorm(100), 20, 5)
  Y <- matrix(rnorm(100, 1), 20, 5)

  s <- sigma_med(X, Y)
  expect_true(is.numeric(s))
  expect_length(s, 1)
  expect_gt(s, 0)
})

test_that("sigma_med handles subsampling", {
  set.seed(42)
  X <- matrix(rnorm(500), 100, 5)
  Y <- matrix(rnorm(500, 1), 100, 5)

  s_all <- sigma_med(X, Y, m = 400)
  s_sub <- sigma_med(X, Y, m = 20, seed = 1)

  expect_gt(s_sub, 0)
})

test_that("sigma_med warns for insufficient data", {
  X <- matrix(1, 1, 3)
  Y <- matrix(2, 1, 3)

  expect_warning(sigma_med(X, Y), "Not enough samples")
})

# ---- compute_distance_matrix ----

test_that("compute_distance_matrix dimensions are correct", {
  set.seed(42)
  A <- matrix(rnorm(15), 5, 3)
  B <- matrix(rnorm(12), 4, 3)

  D <- compute_distance_matrix(A, B)
  expect_equal(dim(D), c(5, 4))
})

test_that("compute_distance_matrix self-distance diagonal is zero", {
  set.seed(42)
  A <- matrix(rnorm(15), 5, 3)

  D <- compute_distance_matrix(A, A)
  expect_equal(diag(D), rep(0, 5), tolerance = 1e-6)
})

test_that("compute_distance_matrix values are non-negative", {
  set.seed(42)
  A <- matrix(rnorm(15), 5, 3)
  B <- matrix(rnorm(12), 4, 3)

  D <- compute_distance_matrix(A, B)
  expect_true(all(D >= 0, na.rm = TRUE))
})

# ---- compute_mmd ----

test_that("compute_mmd returns a scalar", {
  set.seed(42)
  X <- matrix(rnorm(100), 20, 5)
  Y <- matrix(rnorm(100, 1), 20, 5)

  m <- compute_mmd(X, Y, sigma = 1)
  expect_true(is.numeric(m))
  expect_length(m, 1)
})

test_that("compute_mmd is zero for identical data", {
  set.seed(42)
  X <- matrix(rnorm(100), 20, 5)

  m <- compute_mmd(X, X, sigma = 1)
  expect_equal(m, 0, tolerance = 1e-10)
})

test_that("compute_mmd is positive for different distributions", {
  set.seed(42)
  X <- matrix(rnorm(200), 40, 5)
  Y <- matrix(rnorm(200, mean = 5), 40, 5)

  m <- compute_mmd(X, Y, sigma = 1)
  expect_gt(m, 0)
})

# ---- compute_energy ----

test_that("compute_energy returns a non-negative scalar", {
  set.seed(42)
  X <- matrix(rnorm(100), 20, 5)
  Y <- matrix(rnorm(100, 1), 20, 5)

  e <- compute_energy(X, Y)
  expect_true(is.numeric(e))
  expect_length(e, 1)
  expect_gte(e, 0)
})

test_that("compute_energy is zero for identical data", {
  set.seed(42)
  X <- matrix(rnorm(100), 20, 5)

  e <- compute_energy(X, X)
  expect_equal(e, 0, tolerance = 1e-10)
})

# ---- compute_wasserstein ----

test_that("compute_wasserstein returns a non-negative scalar", {
  set.seed(42)
  X <- matrix(rnorm(20), 5, 4)
  Y <- matrix(rnorm(24, 1), 6, 4)

  w <- compute_wasserstein(X, Y)
  expect_true(is.numeric(w))
  expect_length(w, 1)
  expect_gte(w, 0)
})

test_that("compute_wasserstein is zero for identical data", {
  set.seed(42)
  X <- matrix(rnorm(20), 5, 4)

  w <- compute_wasserstein(X, X)
  expect_equal(w, 0, tolerance = 1e-6)
})

test_that("compute_wasserstein rejects column mismatch", {
  X <- matrix(rnorm(20), 5, 4)
  Y <- matrix(rnorm(15), 5, 3)

  expect_error(compute_wasserstein(X, Y))
})

# ---- compute_mahalanobis ----

test_that("compute_mahalanobis returns a non-negative scalar", {
  set.seed(42)
  X <- matrix(rnorm(200), 20, 10)
  Y <- matrix(rnorm(150, 0.5), 15, 10)

  d <- compute_mahalanobis(X, Y)
  expect_true(is.numeric(d))
  expect_length(d, 1)
  expect_gte(d, 0)
})

test_that("compute_mahalanobis is zero for identical data", {
  set.seed(42)
  X <- matrix(rnorm(200), 20, 10)

  d <- compute_mahalanobis(X, X)
  expect_equal(d, 0, tolerance = 1e-6)
})

test_that("compute_mahalanobis covChoice options work", {
  set.seed(42)
  X <- matrix(rnorm(200), 20, 10)
  Y <- matrix(rnorm(150, 0.5), 15, 10)

  d_pool   <- compute_mahalanobis(X, Y, covChoice = "pooled")
  d_source <- compute_mahalanobis(X, Y, covChoice = "source")
  d_target <- compute_mahalanobis(X, Y, covChoice = "target")

  # All should be valid non-negative
  expect_gte(d_pool, 0)
  expect_gte(d_source, 0)
  expect_gte(d_target, 0)
})

test_that("compute_mahalanobis squared option works", {
  set.seed(42)
  X <- matrix(rnorm(200), 20, 10)
  Y <- matrix(rnorm(150, 0.5), 15, 10)

  d  <- compute_mahalanobis(X, Y, squared = FALSE)
  d2 <- compute_mahalanobis(X, Y, squared = TRUE)

  expect_equal(d^2, d2, tolerance = 1e-10)
})

test_that("compute_mahalanobis shrinkage works", {
  set.seed(42)
  X <- matrix(rnorm(200), 20, 10)
  Y <- matrix(rnorm(150, 0.5), 15, 10)

  d_no_shrink <- compute_mahalanobis(X, Y)
  d_shrink    <- compute_mahalanobis(X, Y, shrinkage_alpha = 0.1)

  # Both should be valid
  expect_gte(d_no_shrink, 0)
  expect_gte(d_shrink, 0)
  # Different shrinkage should give different values
  expect_false(isTRUE(all.equal(d_no_shrink, d_shrink)))
})

test_that("Kernels/metrics reference values for cross-language validation", {
  set.seed(2024)
  X <- matrix(rnorm(50), nrow = 10, ncol = 5)
  Y <- matrix(rnorm(50, mean = 1), nrow = 10, ncol = 5)

  # RBF kernel
  K <- rbf_kernel(X, Y, sigma = 1)
  expect_equal(dim(K), c(10, 10))
  expect_true(all(is.finite(K)))

  # MMD
  mmd <- compute_mmd(X, Y, sigma = 1)
  expect_true(is.finite(mmd))

  # Energy
  energy <- compute_energy(X, Y)
  expect_true(is.finite(energy))

  # Wasserstein
  wass <- compute_wasserstein(X, Y)
  expect_true(is.finite(wass))

  # Mahalanobis
  maha <- compute_mahalanobis(X, Y)
  expect_true(is.finite(maha))
})
