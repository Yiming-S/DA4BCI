
#  kernels_metrics.R
#
#  rbf_kernel, sigma_med, compute_distance_matrix
#  compute_mmd, compute_energy, compute_wasserstein, compute_mahalanobis



#' RBF (Gaussian) Kernel Computation
#'
#' @description
#' \code{rbf_kernel} computes the Radial Basis Function (RBF) kernel between
#' two sets of observations \code{x} and \code{y} using
#' \eqn{\exp(-\gamma \|x-y\|^2)} and returns the \code{x}–\code{y} cross-block.
#'
#' @param x Numeric matrix (n_x × p): observations by features.
#' @param y Numeric matrix (n_y × p): must have the same number of columns as \code{x}.
#' @param sigma Positive scalar bandwidth.
#' @param standard_scale Logical. If \code{TRUE} use \eqn{\gamma = 1/(2\sigma^2)};
#'   otherwise \eqn{\gamma = 1/\sigma^2}. Default \code{TRUE}.
#'
#' @details
#' Distances are computed via the identity \eqn{\|x-y\|^2 = \|x\|^2+\|y\|^2-2x^\top y}
#' for numerical stability; tiny negative values are clamped to zero.
#'
#' @return Numeric matrix (n_x × n_y) of RBF kernel values.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' x <- matrix(rnorm(20), 5, 4); y <- matrix(rnorm(24), 6, 4)
#' K <- rbf_kernel(x, y, sigma = 1)
#' dim(K)
#' }
#' @family kernels-metrics
#' @seealso \code{\link{sigma_med}}, \code{\link{compute_mmd}}
#' @export
#'

# rbf_kernel <- function(x, y, sigma) {
#   distx <- as.matrix(dist(rbind(x, y)))
#   kernel <- exp(- (distx / sigma)^2)
#   n <- nrow(x)
#   return(kernel[1:n, -(1:n)])
# }

rbf_kernel <- function(x, y, sigma,
                       standard_scale = TRUE) {
  x <- as.matrix(x); y <- as.matrix(y)
  stopifnot(ncol(x) == ncol(y), is.numeric(sigma), length(sigma) == 1L, sigma > 0)

  xx <- rowSums(x * x)
  yy <- rowSums(y * y)
  D2 <- outer(xx, yy, "+") - 2 * tcrossprod(x, y)  # O(nmd)
  D2 <- pmax(D2, 0)  # guard against tiny negative due to cancellation

  gamma <- if (standard_scale) 1 / (2 * sigma^2) else 1 / (sigma^2)
  exp(-gamma * D2)
}


#' Robust Median-Distance Heuristic for RBF Bandwidth
#'
#' @description
#' \code{sigma_med} concatenates \code{X} and \code{Y}, optionally subsamples rows,
#' computes all pairwise Euclidean distances, and returns their median as a robust
#' bandwidth heuristic.
#'
#' @param X Numeric matrix (n_1 × p).
#' @param Y Numeric matrix (n_2 × p) with the same number of columns as \code{X}.
#' @param m Integer: maximum number of rows used. If total rows exceed \code{m},
#'   a uniform without-replacement subsample is taken. Default \code{400}.
#' @param seed Optional integer seed for reproducibility.
#'
#' @details
#' If the combined sample size is \eqn{\le 2}, returns \code{NA_real_} with a warning.
#' If the median distance is zero (e.g., many duplicates), returns a machine-epsilon
#' positive constant.
#'
#' @return Positive scalar median Euclidean distance; \code{NA_real_} if insufficient data.
#'
#' @examples
#' \dontrun{
#' set.seed(42)
#' X <- matrix(rnorm(100), 20, 5); Y <- matrix(rnorm(100, 1), 20, 5)
#' # Use all rows (N <= m)
#' sigma_all <- sigma_med(X, Y)
#'
#' # Subsample at most 15 rows
#' sigma_sub <- sigma_med(X, Y, m = 15, seed = 1)
#' }
#' @family kernels-metrics
#' @export
#'
sigma_med <- function(X, Y, m = 400, seed = NULL) {
  # Optional reproducibility
  stopifnot(ncol(X) == ncol(Y))
  if (!is.null(seed)) set.seed(seed)

  # Combine domains
  XY <- rbind(X, Y)
  N  <- nrow(XY)

  # Not enough points to form pairwise distances
  if (N <= 2) {
    warning("Not enough samples to compute pairwise distances; returning NA.")
    return(NA_real_)
  }

  # Sub‑sample if necessary
  if (N > m) {
    idx <- sample.int(N, size = m, replace = FALSE)
    XY  <- XY[idx, , drop = FALSE]
  }

  # Median Euclidean distance
  d_med <- median(as.numeric(dist(XY, method = "euclidean")))

  # Avoid zero bandwidth
  if (d_med == 0) d_med <- .Machine$double.eps

  d_med
}



#' Pairwise Euclidean Distance Matrix
#'
#' @description
#' \code{compute_distance_matrix} returns pairwise Euclidean distances between
#' rows of \code{source} and \code{target}.
#'
#' @param source Numeric matrix (n_s × p).
#' @param target Numeric matrix (n_t × p) with the same number of columns.
#' @param eps Numeric tolerance to clamp tiny negative squared distances to zero.
#'
#' @details
#' Uses the identity \eqn{\|x-y\|^2 = \|x\|^2+\|y\|^2-2x^\top y}. Squared distances
#' in \((-{\tt eps},0)\) are set to 0; values below \(-{\tt eps}\) are flagged \code{NA}.
#'
#' @return Numeric matrix (n_s × n_t) of Euclidean distances.
#'
#' @examples
#' \dontrun{
#' A <- matrix(rnorm(15), 5, 3); B <- matrix(rnorm(12), 4, 3)
#' D <- compute_distance_matrix(A, B); dim(D)
#' }
#' @family kernels-metrics
#' @export
#'
compute_distance_matrix <- function(source, target, eps = 1e-12) {
  cross_term <- source %*% t(target)
  source_norms <- rowSums(source^2)
  target_norms <- rowSums(target^2)

  d2 <- outer(source_norms, target_norms, "+") - 2 * cross_term  # squared distances

  # Precision safeguard
  d2[d2 > -eps & d2 < 0] <- 0                   # clamp tiny negatives to zero
  d2[d2 < -eps]         <- NA_real_            # flag unexpected large negatives

  sqrt(d2)
}




#' Squared Maximum Mean Discrepancy (MMD^2) with RBF Kernel
#'
#' @description
#' \code{compute_mmd} estimates \eqn{\mathrm{MMD}^2} between \code{source} and \code{target}
#' using an RBF kernel with bandwidth \code{sigma}.
#'
#' @param source Numeric matrix (m × p).
#' @param target Numeric matrix (n × p) with the same number of columns.
#' @param sigma Positive scalar bandwidth for \code{\link{rbf_kernel}}.
#'
#' @details
#' \eqn{\mathrm{MMD}^2 = m^{-2}\!\sum K_{ss} + n^{-2}\!\sum K_{tt} - 2(mn)^{-1}\!\sum K_{st}}.
#'
#' @return Scalar \eqn{\mathrm{MMD}^2}.
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(100), 20, 5); Y <- matrix(rnorm(100, 1), 20, 5)
#' compute_mmd(X, Y, sigma = 1)
#' }
#' @family kernels-metrics
#' @seealso \code{\link{rbf_kernel}}, \code{\link{sigma_med}}
#' @export
#'
compute_mmd <- function(source, target, sigma) {
  Kss <- rbf_kernel(source, source, sigma)
  Ktt <- rbf_kernel(target, target, sigma)
  Kst <- rbf_kernel(source, target, sigma)

  m <- nrow(source)
  n <- nrow(target)

  mmd <- sum(Kss) / (m * m) + sum(Ktt) / (n * n) - 2 * sum(Kst) / (m * n)
  return(mmd)
}



#' Energy Distance Between Empirical Distributions
#'
#' @description
#' \code{compute_energy} computes the Energy Distance between \code{source} and \code{target}
#' via empirical pairwise Euclidean distances.
#'
#' @param source Numeric matrix (n_s × p).
#' @param target Numeric matrix (n_t × p) with the same number of columns.
#'
#' @details
#' Estimates \eqn{D_E = \sqrt{2\,\mathbb{E}\|X-Y\| - \mathbb{E}\|X-X'\| - \mathbb{E}\|Y-Y'\|}},
#' replacing expectations by sample averages; negative under-root values due to
#' numerical error are clamped to zero.
#'
#' @return Scalar Energy Distance (nonnegative).
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(100), 20, 5); Y <- matrix(rnorm(100, 1), 20, 5)
#' compute_energy(X, Y)
#' }
#' @family kernels-metrics
#' @seealso \code{\link{compute_mmd}}, \code{\link{compute_wasserstein}}
#' @export
#'
compute_energy <- function(source, target) {
  # Intra-domain pairwise distances
  ds   <- compute_distance_matrix(source, source)
  dt   <- compute_distance_matrix(target, target)

  # Inter-domain pairwise distances
  d_st <- compute_distance_matrix(source, target)

  # Empirical estimator (Székely et al.)
  ed2 <- 2 * mean(d_st, na.rm = TRUE) - mean(ds, na.rm = TRUE) - mean(dt, na.rm = TRUE)

  # Guard against tiny negative values due to numerical error
  sqrt(pmax(ed2, 0))
}



#' Empirical 1-Wasserstein Distance (Uniform Weights)
#'
#' @description
#' \code{compute_wasserstein} computes the empirical 1-Wasserstein distance between
#' \code{source} and \code{target} using Euclidean costs and a uniform coupling solved
#' by \code{transport::transport}.
#'
#' @param source Numeric matrix (n_s × p).
#' @param target Numeric matrix (n_t × p) with the same number of columns.
#'
#' @details
#' The cost matrix is \code{\link{compute_distance_matrix}(source, target)}; the optimal
#' plan is solved with method \code{"revsimplex"} in \pkg{transport}.
#'
#' @return Scalar 1-Wasserstein distance.
#'
#' @examples
#' \dontrun{
#' A <- matrix(rnorm(20), 5, 4); B <- matrix(rnorm(24, 1), 6, 4)
#' compute_wasserstein(A, B)
#' }
#' @family kernels-metrics
#' @seealso \code{\link{compute_energy}}, \code{\link{compute_mmd}}
#' @export
#'
compute_wasserstein <- function(source, target) {
  stopifnot(is.matrix(source), is.matrix(target))
  if (ncol(source) != ncol(target)) {
    stop("Source and target must have the same number of features (columns).")
  }

  n_s <- nrow(source)
  n_t <- nrow(target)
  p <- rep(1 / n_s, n_s)
  q <- rep(1 / n_t, n_t)

  cost_matrix <- compute_distance_matrix(source, target)
  plan <- transport::transport(p, q, cost_matrix, method = "revsimplex")
  sum(plan$mass * cost_matrix[cbind(plan$from, plan$to)])
}



#' Mahalanobis distance between two datasets (means with pooled/selected covariance)
#'
#' @description
#' compute_mahalanobis measures the distance between the source and target
#' mean vectors under a Mahalanobis metric induced by a chosen covariance
#' (pooled/source/target). It includes optional shrinkage and a small ridge
#' for numerical stability in medium/high dimensions.
#'
#' @param source Numeric matrix (n_s × p): source samples.
#' @param target Numeric matrix (n_t × p): target samples.
#' @param covChoice Character, one of c("pooled","source","target"); default "pooled".
#' @param shrinkage_alpha Optional scalar in [0,1]; shrinkage toward spherical target
#'   (tr(S)/p) * I. Use small values like 0.05 ~ 0.2 if n is small or p is large.
#' @param ridge Nonnegative scalar ridge added as (ridge * tr(S)/p) * I. Default 1e-6.
#' @param squared Logical; if TRUE, return squared distance. Default FALSE.
#'
#' @return A single numeric distance (Mahalanobis) between the two domains.
#' @export
#'
#' @examples
#' \dontrun{
#'   set.seed(1)
#'   X <- matrix(rnorm(200), 20, 10)
#'   Y <- matrix(rnorm(150, 0.5), 15, 10)
#'   compute_mahalanobis(X, Y)                         # pooled covariance
#'   compute_mahalanobis(X, Y, covChoice = "source")   # source covariance
#'   compute_mahalanobis(X, Y, shrinkage_alpha = 0.1)  # mild shrinkage
#' }
compute_mahalanobis <- function(source, target,
                                covChoice = c("pooled","source","target"),
                                shrinkage_alpha = NULL,
                                ridge = 1e-6,
                                squared = FALSE) {
  # Basic checks
  stopifnot(is.matrix(source), is.matrix(target))
  if (ncol(source) != ncol(target)) {
    stop("source/target must have same number of columns")
  }

  # Means and covariances
  mux <- colMeans(source)
  muy <- colMeans(target)
  Sx  <- stats::cov(source)
  Sy  <- stats::cov(target)

  covChoice <- match.arg(covChoice)
  nx <- nrow(source); ny <- nrow(target); p <- ncol(source)

  # Choose covariance for the metric
  if (covChoice == "pooled") {
    if (nx + ny - 2 <= 0) stop("Not enough samples for pooled covariance.")
    S <- ((nx - 1) * Sx + (ny - 1) * Sy) / (nx + ny - 2)
  } else if (covChoice == "source") {
    S <- Sx
  } else {
    S <- Sy
  }

  # Symmetrize for numerical safety
  S <- 0.5 * (S + t(S))

  # Optional shrinkage toward spherical target: (1 - a) S + a * (tr(S)/p) I
  if (!is.null(shrinkage_alpha)) {
    if (!is.numeric(shrinkage_alpha) || length(shrinkage_alpha) != 1L ||
        shrinkage_alpha < 0 || shrinkage_alpha > 1) {
      stop("shrinkage_alpha must be in [0,1].")
    }
    trp <- sum(diag(S)) / p
    S <- (1 - shrinkage_alpha) * S + shrinkage_alpha * trp * diag(p)
  }

  # Small ridge scaled by average variance to ensure positive definiteness
  if (!is.null(ridge) && ridge > 0) {
    trp <- sum(diag(S)) / p
    S <- S + ridge * trp * diag(p)
  }

  # Stable inversion via eigen decomposition (SPD guard)
  eig <- eigen(S, symmetric = TRUE)
  eps <- .Machine$double.eps^0.5
  eig$values[eig$values < eps] <- eps
  SInv <- eig$vectors %*% (diag(1 / eig$values)) %*% t(eig$vectors)

  # Mahalanobis between domain means
  d <- mux - muy
  d2 <- as.numeric(t(d) %*% SInv %*% d)
  if (isTRUE(squared)) d2 else sqrt(d2)
}

















