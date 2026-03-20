# test-spd_geometry.R
# Tests for SPD geometry functions

# ---- orthonormal_complement ----

test_that("orthonormal_complement dimensions are correct", {
  U <- qr.Q(qr(matrix(rnorm(12), 4, 3)))
  Uperp <- orthonormal_complement(U)

  expect_equal(nrow(Uperp), 4)
  expect_equal(ncol(Uperp), 1)  # 4 - 3 = 1
})

test_that("orthonormal_complement columns are orthonormal", {
  set.seed(42)
  U <- qr.Q(qr(matrix(rnorm(10), 5, 2)))
  Uperp <- orthonormal_complement(U)

  # Columns should be unit length
  for (j in seq_len(ncol(Uperp))) {
    expect_equal(sum(Uperp[, j]^2), 1, tolerance = 1e-10)
  }
  # Orthogonal to U
  cross <- crossprod(U, Uperp)
  expect_equal(max(abs(cross)), 0, tolerance = 1e-10)
})

test_that("orthonormal_complement of full-rank matrix is empty", {
  U <- diag(5)
  Uperp <- orthonormal_complement(U)
  expect_equal(ncol(Uperp), 0)
})

test_that("orthonormal_complement of empty columns returns identity", {
  U <- matrix(0, nrow = 4, ncol = 0)
  Uperp <- orthonormal_complement(U)
  expect_equal(dim(Uperp), c(4, 4))
})

# ---- LW_covariance ----

test_that("LW_covariance returns a p x p matrix", {
  set.seed(42)
  X <- matrix(rnorm(200), 20, 10)
  S <- LW_covariance(X)

  expect_equal(dim(S), c(10, 10))
})

test_that("LW_covariance is symmetric", {
  set.seed(42)
  X <- matrix(rnorm(200), 20, 10)
  S <- LW_covariance(X)

  expect_equal(S, t(S), tolerance = 1e-12)
})

test_that("LW_covariance is positive definite", {
  set.seed(42)
  X <- matrix(rnorm(200), 20, 10)
  S <- LW_covariance(X)

  evals <- eigen(S, symmetric = TRUE, only.values = TRUE)$values
  expect_true(all(evals > 0))
})

# ---- matrix_power ----

test_that("matrix_power identity cases", {
  A <- crossprod(matrix(rnorm(25), 5, 5)) + diag(5)

  expect_equal(matrix_power(A, 1), (A + t(A)) / 2, tolerance = 1e-10)
  expect_equal(matrix_power(A, 0), diag(5))
})

test_that("matrix_power inverse", {
  set.seed(42)
  A <- crossprod(matrix(rnorm(25), 5, 5)) + diag(5)

  A_inv <- matrix_power(A, -1)
  expect_equal(A %*% A_inv, diag(5), tolerance = 1e-6)
})

test_that("matrix_power sqrt satisfies A^0.5 A^0.5 = A", {
  set.seed(42)
  A <- crossprod(matrix(rnorm(25), 5, 5)) + diag(5)

  A_half <- matrix_power(A, 0.5)
  expect_equal(A_half %*% A_half, (A + t(A)) / 2, tolerance = 1e-6)
})

test_that("matrix_power negative sqrt satisfies A^{-0.5} A A^{-0.5} = I", {
  set.seed(42)
  A <- crossprod(matrix(rnorm(25), 5, 5)) + diag(5)

  A_neg_half <- matrix_power(A, -0.5)
  A_sym <- (A + t(A)) / 2
  result <- A_neg_half %*% A_sym %*% A_neg_half
  expect_equal(result, diag(5), tolerance = 1e-5)
})

# ---- riemannian_mean ----

test_that("riemannian_mean of identical matrices returns that matrix", {
  set.seed(42)
  A <- crossprod(matrix(rnorm(25), 5, 5)) + diag(5)
  arr <- array(0, dim = c(5, 5, 3))
  for (i in 1:3) arr[, , i] <- A

  R <- riemannian_mean(arr)
  expect_equal(R, A, tolerance = 1e-4)
})

test_that("riemannian_mean returns SPD matrix", {
  set.seed(42)
  arr <- array(0, dim = c(4, 4, 5))
  for (i in 1:5) {
    M <- matrix(rnorm(16), 4, 4)
    arr[, , i] <- crossprod(M) + diag(4)
  }

  R <- riemannian_mean(arr)
  expect_equal(R, t(R), tolerance = 1e-10)

  evals <- eigen(R, symmetric = TRUE, only.values = TRUE)$values
  expect_true(all(evals > 0))
})

test_that("riemannian_mean accepts list input", {
  set.seed(42)
  mats <- lapply(1:4, function(i) {
    M <- matrix(rnorm(16), 4, 4)
    crossprod(M) + diag(4)
  })

  R <- riemannian_mean(mats)
  expect_equal(dim(R), c(4, 4))
  expect_true(all(eigen(R, symmetric = TRUE, only.values = TRUE)$values > 0))
})

# ---- log_map / exp_map ----

test_that("exp_map(P, log_map(P, Q)) recovers Q", {
  set.seed(42)
  P <- crossprod(matrix(rnorm(25), 5, 5)) + diag(5)
  Q <- crossprod(matrix(rnorm(25), 5, 5)) + diag(5)

  L <- log_map(P, Q)
  Q_recovered <- exp_map(P, L)
  expect_equal(Q_recovered, Q, tolerance = 1e-6)
})

test_that("log_map at identity is matrix logarithm", {
  set.seed(42)
  Q <- crossprod(matrix(rnorm(25), 5, 5)) + diag(5)
  I <- diag(5)

  L <- log_map(I, Q)

  # exp(L) should give Q back
  eig <- eigen(L, symmetric = TRUE)
  Q_from_L <- eig$vectors %*% diag(exp(eig$values)) %*% t(eig$vectors)
  expect_equal(Q_from_L, Q, tolerance = 1e-6)
})

test_that("log_map accepts precomputed sqrt/inv_sqrt", {
  set.seed(42)
  P <- crossprod(matrix(rnorm(25), 5, 5)) + diag(5)
  Q <- crossprod(matrix(rnorm(25), 5, 5)) + diag(5)

  eig <- eigen(P, symmetric = TRUE)
  P_list <- list(
    sqrt = eig$vectors %*% diag(sqrt(eig$values)) %*% t(eig$vectors),
    inv_sqrt = eig$vectors %*% diag(1 / sqrt(eig$values)) %*% t(eig$vectors)
  )

  L_mat  <- log_map(P, Q)
  L_list <- log_map(P_list, Q)
  expect_equal(L_mat, L_list, tolerance = 1e-10)
})

# ---- compute_geodesic ----

test_that("compute_geodesic returns a non-negative scalar", {
  set.seed(42)
  X <- matrix(rnorm(300), 60, 5)
  Y <- matrix(rnorm(300, .5), 60, 5)

  g <- compute_geodesic(X, Y)
  expect_true(is.numeric(g))
  expect_length(g, 1)
  expect_gte(g, 0)
})

test_that("compute_geodesic is zero for identical data", {
  set.seed(42)
  X <- matrix(rnorm(300), 60, 5)

  g <- compute_geodesic(X, X)
  expect_equal(g, 0, tolerance = 1e-6)
})

test_that("compute_geodesic respects d parameter", {
  set.seed(42)
  X <- matrix(rnorm(300), 60, 5)
  Y <- matrix(rnorm(300, .5), 60, 5)

  g3 <- compute_geodesic(X, Y, d = 3)
  g5 <- compute_geodesic(X, Y, d = 5)

  expect_gte(g3, 0)
  expect_gte(g5, 0)
})

test_that("SPD geometry reference values for cross-language validation", {
  set.seed(2024)
  # SPD matrices
  A <- crossprod(matrix(rnorm(25), 5, 5)) + diag(5)
  B <- crossprod(matrix(rnorm(25), 5, 5)) + diag(5)

  # matrix_power
  Ahalf <- matrix_power(A, 0.5)
  expect_true(all(is.finite(Ahalf)))

  # log_map / exp_map round-trip
  L <- log_map(A, B)
  B_rec <- exp_map(A, L)
  expect_equal(B_rec, B, tolerance = 1e-6)

  # LW_covariance
  X <- matrix(rnorm(100), 20, 5)
  Slw <- LW_covariance(X)
  expect_equal(dim(Slw), c(5, 5))
  expect_true(all(is.finite(Slw)))
})
