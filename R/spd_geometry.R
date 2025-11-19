
# spd_geometry.R
#
#  orthonormal_complement
#  LW_covariance, matrix_power
#  riemannian_mean, log_map, exp_map, align_riemannian_transport
#  compute_geodesic

#' Orthonormal Complement of a Column Space
#'
#' @description
#' \code{orthonormal_complement} returns an orthonormal basis of the subspace
#' orthogonal to \code{U} in \eqn{\mathbb{R}^d}.
#'
#' @param U Numeric matrix (d × k) with (approximately) orthonormal columns.
#' @param tol Numeric tolerance for rank-revealing QR.
#'
#' @details
#' Uses \code{\link[base]{qr}}; if \eqn{\mathrm{rank}(U)=r<d}, the last \eqn{d-r}
#' columns of the complete Q form the complement. If \eqn{r \ge d}, returns a
#' \eqn{d \times 0} matrix.
#'
#' @return Numeric matrix (d × (d - r)) with orthonormal columns.
#'
#' @examples
#' \dontrun{
#' U <- qr.Q(qr(matrix(rnorm(12), 4, 3)))
#' Uperp <- orthonormal_complement(U); dim(Uperp)
#' }
#' @family spd-geometry
#' @export
#'
orthonormal_complement <- function(U, tol = 1e-8) {

  # Basic input checks
  if (!is.matrix(U)) stop("U must be a numeric matrix.")
  if (!is.numeric(U)) stop("U must be numeric.")

  d <- nrow(U)
  k <- ncol(U)

  if (d == 0L) stop("U must have at least one row.")
  if (k == 0L) return(diag(d))
  if (d < k) stop("U must have at least as many rows as columns (d >= k).")

  # Rank-revealing QR decomposition of U
  # qr() determines numerical rank using the provided tolerance
  qr_U <- qr(U, tol = tol)
  r <- qr_U$rank

  # If U spans the whole space, the complement has zero columns
  if (r >= d) {
    return(matrix(0, nrow = d, ncol = 0))
  }

  # Obtain a complete orthonormal basis Q (d x d)
  Q_full <- qr.Q(qr_U, complete = TRUE)

  # Extract the last d - r columns as the orthonormal complement
  Q_perp <- Q_full[, (r + 1L):d, drop = FALSE]

  qr_perp <- qr(Q_perp, tol = tol)
  Q_perp <- qr.Q(qr_perp)

  return(Q_perp)
}


#' Ledoit–Wolf Shrinkage Covariance (to Identity)
#'
#' @description
#' \code{LW_covariance} estimates a covariance matrix with Ledoit–Wolf shrinkage
#' toward the identity, robust for high-dimensional/small-sample settings.
#'
#' @param x Numeric matrix (n × p): rows are observations.
#'
#' @details
#' Centers columns, forms sample covariance, estimates shrinkage intensity, and
#' returns the shrunk covariance.
#'
#' @return Numeric (p × p) covariance matrix.
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(200), 20, 10)
#' S <- LW_covariance(X); dim(S)
#' }
#' @family spd-geometry
#'
LW_covariance <- function(x) {
  n <- nrow(x)  # Number of observations
  p <- ncol(x)  # Number of variables

  # Center the data
  x <- x - matrix(colMeans(x), n, p, TRUE)

  # Compute the sample covariance matrix
  S <- crossprod(x) / n

  # Mean of the diagonal elements of the sample covariance
  m <- mean(diag(S))

  # Calculate d2 and bbar2 for shrinkage parameter lambda
  d2 <- sum((S - diag(m, p))^2) / p
  bbar2 <- (sum(rowSums(x^2)^2) - n * sum(S^2) ) / (p * n^2)
  b2 <- min(bbar2, d2)

  # Compute the Ledoit-Wolf shrinkage covariance matrix
  rho <- b2 / d2
  S <- (1-rho) * S
  diag(S) <- diag(S) + rho * m
  return(S)
}



#' Matrix Power for Symmetric Positive Definite Matrices
#'
#' @description
#' \code{matrix_power} computes \eqn{A^{\mathrm{pow}}} for SPD \code{A} via
#' symmetric eigendecomposition.
#'
#' @param A Numeric symmetric positive definite matrix (p × p).
#' @param pow Numeric exponent (e.g., 1, 0, -1, ±1/2, etc.).
#'
#' @details
#' Uses eigenvalues clamped below by \code{1e-8} to guard numerical issues; handles
#' common shortcuts for \code{pow} in \{1, 0, -1\}.
#'
#' @return Numeric (p × p) matrix \eqn{A^{\mathrm{pow}}}.
#'
#' @examples
#' \dontrun{
#' A <- crossprod(matrix(rnorm(25), 5, 5)) + diag(5)
#' Ahalf <- matrix_power(A, 0.5)
#' }
#' @family spd-geometry
#'

matrix_power <- function(A, pow, eig_eps   = 1e-6,
                        ridge_eps = 1e-6,
                        max_retry = 3L,
                        verbose   = FALSE) {
  # Symmetrize and trivial powers
  A <- (A + t(A)) / 2
  n <- nrow(A)

  if (pow == 1) return(A)
  if (pow == 0) return(diag(n))

  # Non-finite check
  if (any(!is.finite(A))) {
    if (verbose) message("[matrix_power] Non-finite A, returning identity.")
    return(diag(n))
  }

  # Fast path for inverse (pow = -1): Strict PD check via Cholesky
  if (pow == -1) {
    # chol() fails if A is not PD, automatically forcing fallback to spectral repair
    R <- try(chol(A), silent = TRUE)
    if (!inherits(R, "try-error")) {
      if (verbose) message("[matrix_power] Used Cholesky (strictly PD) for inverse.")
      return(chol2inv(R))
    }
    if (verbose) message("[matrix_power] Cholesky failed (Non-PD), fallback to spectral.")
  }

  # Spectral decomposition with iterative ridge
  ridge  <- ridge_eps
  eig_ok <- FALSE
  ee     <- NULL

  for (attempt in seq_len(max_retry)) {
    M  <- A + diag(ridge, n)
    ee <- try(eigen(M, symmetric = TRUE), silent = TRUE)
    if (!inherits(ee, "try-error")) {
      eig_ok <- TRUE
      break
    }
    ridge <- ridge * 10
    if (verbose) {
      message(sprintf("[matrix_power] eigen failed, ridge=%.3e (attempt %d)",
                      ridge, attempt + 1L))
    }
  }

  # Still failed -> identity
  if (!eig_ok) {
    if (verbose) message("[matrix_power] eigen failed after retries, returning identity.")
    return(diag(n))
  }

  # Clamp eigenvalues and reconstruct
  vals <- Re(ee$values)
  vals[!is.finite(vals)] <- eig_eps # Kept as requested "small defense line"
  vals <- pmax(vals, eig_eps)

  vals_pow <- if (pow == -1) 1 / vals else vals ^ pow
  V        <- ee$vectors

  Vscaled <- V * rep(sqrt(vals_pow), each = n)
  tcrossprod(Vscaled)
}


#' Riemannian (Affine-Invariant) Mean of SPD Matrices
#'
#' @description
#' \code{riemannian_mean} computes the affine-invariant Riemannian mean of a set
#' of SPD matrices by iterative log–exp mapping.
#'
#' @param cov_matrices 3D array (p × p × m) or list of (p × p) SPD matrices.
#' @param max_iterations Integer maximum iterations. Default \code{500}.
#' @param epsilon Convergence tolerance on Frobenius norm. Default \code{1e-5}.
#'
#' @details
#' Initializes with elementwise mean; iterates by mapping each matrix to the tangent
#' space at current mean, averaging, and mapping back via the exponential map.
#'
#' @return Numeric (p × p) SPD matrix: the Riemannian mean.
#'
#' @family spd-geometry
#'
riemannian_mean <- function(cov_matrices, max_iterations = 500,
                            epsilon = 1e-5) {
  if (is.list(cov_matrices))
    cov_matrices <- simplify2array(cov_matrices)
  # Number of covariance matrices
  m <- dim(cov_matrices)[3]
  n <- dim(cov_matrices)[1]

  # Step 1: Initialize the Riemannian mean
  # Use the element-wise mean of the covariance matrices as the initial estimate
  P_omega <- rowMeans(cov_matrices, dims = 2L)

  # Step 2: Iterate to refine the Riemannian mean
  for (iteration in 1:max_iterations) {
    # Step 2.1: Compute the square root and inverse square root of the mean matrix (P_omega)
    eig <- eigen(P_omega, symmetric = TRUE)
    sqrt_P_omega <- eig$vectors %*% (t(eig$vectors) * sqrt(eig$values))
    inv_sqrt_P_omega <- eig$vectors %*% (t(eig$vectors) / sqrt(eig$values))

    # Step 2.2: Compute the average in the tangent space (logarithmic map)
    S <- matrix(0, n, n) # Initialize tangent space mean
    for (i in 1:m) {
      # Map each covariance matrix to the tangent space
      eig <- eigen(inv_sqrt_P_omega %*% cov_matrices[, , i] %*%
                     inv_sqrt_P_omega, symmetric = TRUE)

      # Accumulate the log map
      S <- S + (eig$vectors %*% (t(eig$vectors) * log(eig$values)))
    }
    S <- S / m # Average the log-mapped matrices

    # Note: S does not require further transformation by sqrt_P_omega
    # because it will be multiplied immediately after by inv_sqrt_P_omega
    # in the exponential map (the two operations cancel each other)
    PS <- P_omega %*% S # needed to check convergence

    # Step 2.3: Map back to the SPD matrix space (exponential map)
    eig <- eigen(S, symmetric = TRUE)
    P_omega <- sqrt_P_omega %*% eig$vectors %*%
      (t(eig$vectors) * exp(eig$values)) %*% sqrt_P_omega

    # Step 2.4: Check for convergence
    # Convergence is measured by the Frobenius norm of the product of P_omega and S
    if (sqrt(sum(PS * t(PS))) < epsilon) break
  }

  # Step 3: Return the Riemannian mean
  return(P_omega)
}



#' Riemannian Logarithmic Map at an SPD Point
#'
#' @description
#' \code{log_map} maps SPD matrix \code{P_i} to the tangent space at \code{P_omega}
#' under the affine-invariant metric.
#'
#' @param P_omega Either an SPD matrix (p × p) or a list with precomputed
#'   components \code{sqrt} and \code{inv_sqrt}.
#' @param P_i SPD matrix (p × p) to be mapped.
#'
#' @return Numeric (p × p) symmetric matrix in the tangent space at \code{P_omega}.
#'
#' @examples
#' \dontrun{
#' A <- crossprod(matrix(rnorm(25),5,5)) + diag(5)
#' B <- crossprod(matrix(rnorm(25),5,5)) + diag(5)
#' L <- log_map(A, B)
#' }
#' @family spd-geometry
#'
log_map <- function(P_omega, P_i) {
  # Project P_i onto the tangent space at P_omega
  if (is.list(P_omega)) {
    sqrt_P_omega <- P_omega[["sqrt"]]
    inv_sqrt_P_omega <- P_omega[["inv_sqrt"]]
  } else {
    eig <- eigen(P_omega, symmetric = TRUE)
    sqrt_P_omega <- eig$vectors %*% diag(sqrt(eig$values)) %*% t(eig$vectors)
    inv_sqrt_P_omega <- eig$vectors %*% diag(1 / sqrt(eig$values)) %*% t(eig$vectors)
  }

  # Map P_i into the tangent space of P_omega
  w <- inv_sqrt_P_omega %*% P_i %*% inv_sqrt_P_omega
  eig_w <- eigen(w, symmetric = TRUE)
  log_w <- eig_w$vectors %*% diag(log(eig_w$values)) %*% t(eig_w$vectors)

  # Project back to the original space
  result <- sqrt_P_omega %*% log_w %*% sqrt_P_omega
  return(result)
}



#' Riemannian Exponential Map at an SPD Point
#'
#' @description
#' \code{exp_map} maps a tangent vector \code{S_i} at \code{P_omega} back to the
#' SPD manifold under the affine-invariant metric.
#'
#' @param P_omega Either an SPD matrix (p × p) or a list with \code{sqrt} and \code{inv_sqrt}.
#' @param S_i Symmetric (p × p) tangent matrix.
#'
#' @return Numeric (p × p) SPD matrix on the manifold.
#'
#' @examples
#' \dontrun{
#' A <- crossprod(matrix(rnorm(25),5,5)) + diag(5)
#' L <- diag(5) * 0.1
#' E <- exp_map(A, L)
#' }
#' @family spd-geometry
#'
exp_map <- function(P_omega, S_i) {
  # Extract or calculate the square root and inverse square root of P_omega
  if (is.list(P_omega)) {
    sqrt_P_omega <- P_omega[["sqrt"]]
    inv_sqrt_P_omega <- P_omega[["inv_sqrt"]]
  } else {
    eig <- eigen(P_omega, symmetric = TRUE)
    sqrt_P_omega <- eig$vectors %*% diag(sqrt(eig$values)) %*% t(eig$vectors)
    inv_sqrt_P_omega <- eig$vectors %*% diag(1 / sqrt(eig$values)) %*% t(eig$vectors)
  }

  # Project S_i into the tangent space of P_omega
  w <- inv_sqrt_P_omega %*% S_i %*% inv_sqrt_P_omega
  eig_w <- eigen(w, symmetric = TRUE)

  # Compute the matrix exponential in the tangent space
  exp_w <- eig_w$vectors %*% diag(exp(eig_w$values)) %*% t(eig_w$vectors)

  # Map back to the original space
  result <- sqrt_P_omega %*% exp_w %*% sqrt_P_omega
  return(result)
}



#' Align SPD Sets by Riemannian Log–Exp Transport
#'
#' @description
#' \code{align_riemannian_transport} aligns source covariances to target geometry
#' by mapping each source SPD through the log map at the source mean and exp map
#' at the target mean.
#'
#' @param cov_S List or array of SPD matrices (source).
#' @param cov_T List or array of SPD matrices (target).
#'
#' @return List of aligned source SPD matrices in target geometry.
#'
#' @examples
#' \dontrun{
#' # cov_S, cov_T as lists of SPD matrices
#' out <- align_riemannian_transport(cov_S, cov_T)
#' }
#' @family spd-geometry
#'
align_riemannian_transport <- function(cov_S, cov_T) {
  if (length(cov_S) == 1) cov_S <- c(cov_S, cov_S)
  if (length(cov_T) == 1) cov_T <- c(cov_T, cov_T)

  mean_S <- riemannian_mean(cov_S)
  mean_T <- riemannian_mean(cov_T)

  lapply(cov_S, function(C) {
    exp_map(mean_T, log_map(mean_S, C))
  })
}



#' Geodesic Distance Between Column Subspaces (Grassmann)
#'
#' @description
#' \code{compute_geodesic} orthonormalizes column spaces via QR and returns the
#' Grassmann geodesic distance from principal angles.
#'
#' @param source Numeric matrix (n_s × p).
#' @param target Numeric matrix (n_t × p) with the same number of columns.
#' @param d Optional integer \eqn{\le p}: subspace dimension; default uses minimal rank.
#'
#' @details
#' Columns are centered before QR. Principal angles come from SVD of \eqn{U^\top V};
#' distance is the \eqn{\ell_2} norm of angles.
#'
#' @return Nonnegative scalar geodesic distance.
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(300), 60, 5); Y <- matrix(rnorm(300, .5), 60, 5)
#' compute_geodesic(X, Y); compute_geodesic(X, Y, d = 3)
#' }
#' @family spd-geometry
#' @export
#'
compute_geodesic <- function(source, target, d = NULL) {
  stopifnot(is.matrix(source), is.matrix(target),
            ncol(source) == ncol(target))

  p <- ncol(source)
  # Decide subspace dimension
  if (is.null(d)) {
    d_source <- qr(source)$rank
    d_target <- qr(target)$rank
    d        <- min(d_source, d_target)
  } else {
    d <- as.integer(d)
    if (d < 1L || d > p)
      stop("'d' must be between 1 and number of columns")
  }

  # Center columns and orthonormalise via QR
  orthonorm_basis <- function(X, k) {
    X_centered <- scale(X, center = TRUE, scale = FALSE)
    Q <- qr.Q(qr(X_centered))
    Q[, seq_len(min(k, ncol(Q))), drop = FALSE]        # p × k orthonormal matrix
  }

  U <- orthonorm_basis(source, d)
  V <- orthonorm_basis(target, d)

  # Principal angles
  svd_uv <- svd(crossprod(U, V), nu = 0, nv = 0)       # SVD of UᵀV
  cos_t  <- pmin(pmax(svd_uv$d, -1), 1)               # clamp to valid range
  theta  <- acos(cos_t)

  # Geodesic distance
  sqrt(sum(theta^2))
}













