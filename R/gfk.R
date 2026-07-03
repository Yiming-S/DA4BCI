
#####################################
#' Geodesic Flow Kernel (GFK) for Domain Adaptation
#'
#' @description
#' The `domain_adaptation_gfk` function computes a Geodesic Flow Kernel (GFK)
#' between source and target domains by modeling a continuous geodesic path
#' on the Grassmann manifold. It first extracts principal subspaces for both
#' domains via PCA, then integrates along the path between these two subspaces
#' to construct a kernel matrix \code{G}. Both source and target data are
#' subsequently projected by \code{G}, reducing domain discrepancies.
#'
#' @param source_data A numeric matrix representing the source domain,
#'   with rows as observations and columns as features.
#' @param target_data A numeric matrix representing the target domain,
#'   with rows as observations and columns as features, having the same number
#'   of columns as \code{source_data}.
#' @param dim_subspace An integer specifying the dimensionality of the principal
#'   subspace retained for both source and target. Defaults to 10, but is capped
#'   by \code{min(ncol(source_data), ncol(target_data))}.
#'
#' @return A list containing:
#' \describe{
#'   \item{\code{weighted_source_data}}{The source data projected by \eqn{G^{1/2}}, so
#'     that Euclidean inner products in the transformed space equal the GFK kernel.}
#'   \item{\code{target_data}}{The target data projected by the same \eqn{G^{1/2}}.}
#'   \item{\code{G}}{A \eqn{p \times p} geodesic flow kernel built from the principal
#'     angles between the source and target PCA subspaces.}
#' }
#'
#' @details
#' \strong{Geodesic Flow Kernel (GFK)} constructs a smooth interpolation on the
#' Grassmann manifold between the subspaces spanned by the top principal components
#' of \code{source_data} and \code{target_data}. By integrating along this geodesic,
#' the method yields a kernel \eqn{G} that can map source and target data into an
#' aligned feature space. This helps mitigate distribution shifts in domain adaptation
#' scenarios, particularly in applications where linear subspace methods are insufficient.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' source_mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
#' target_mat <- matrix(rnorm(200, mean = 2), nrow = 20, ncol = 10)
#'
#' # Apply GFK with a subspace of dimension 5
#' gfk_result <- domain_adaptation_gfk(source_mat, target_mat, dim_subspace = 5)
#' aligned_source <- gfk_result$weighted_source_data
#' aligned_target <- gfk_result$target_data
#'
#' # Inspect the resulting kernel
#' G_matrix <- gfk_result$G
#' dim(G_matrix)  # should match the feature dimension
#' }
#'
#' @export
#####################################


domain_adaptation_gfk <- function(source_data, target_data, dim_subspace = 10) {

  # Scaled PCA subspaces. Population-std scaling matches the Python twin's
  # sklearn StandardScaler; a uniform (per-n) scale factor does not change the
  # subspace, so this also matches prcomp(scale. = TRUE) up to that constant.
  scale_pop <- function(X) {
    mu  <- colMeans(X)
    Xc  <- sweep(X, 2, mu, "-")
    sdv <- sqrt(colMeans(Xc^2))
    sdv[sdv < 1e-12] <- 1
    sweep(Xc, 2, sdv, "/")
  }

  p  <- ncol(source_data)
  Vs <- svd(scale_pop(source_data), nu = 0)$v      # (p x min(n_s, p))
  Vt <- svd(scale_pop(target_data), nu = 0)$v      # (p x min(n_t, p))

  # Clamp k to a dimension both domains actually span, so few-trial inputs do
  # not crash. For a meaningful flow use k < p.
  k  <- min(dim_subspace, p, ncol(Vs), ncol(Vt))
  Ps <- Vs[, seq_len(k), drop = FALSE]             # (p x k) source subspace
  Pt <- Vt[, seq_len(k), drop = FALSE]             # (p x k) target subspace
  Rs <- orthonormal_complement(Ps)                 # (p x (p - k)) source complement

  # Closed-form geodesic-flow integral diagonals for principal angles theta,
  # with the correct theta -> 0 limits (lam1 -> 2, lam2 -> 0, lam3 -> 0).
  flow_diagonals <- function(theta) {
    eps   <- 1e-12
    two_t <- 2 * theta
    safe  <- ifelse(theta < eps, 1, two_t)
    ratio <- ifelse(theta < eps, 1, sin(two_t) / safe)
    cross <- ifelse(theta < eps, 0, (cos(two_t) - 1) / safe)
    list(lam1 = 1 + ratio, lam2 = cross, lam3 = 1 - ratio)
  }

  # Principal angles between the source and target SUBSPACES (k x k). The old
  # code used t(U0) %*% U1 on the full orthonormal bases, which is always
  # orthogonal -> the angles were structurally trivial and G was a no-op or noise.
  sv    <- svd(crossprod(Ps, Pt))                  # SVD of t(Ps) %*% Pt
  U1    <- sv$u
  gamma <- sv$d
  V1    <- sv$v
  theta <- acos(pmin(pmax(gamma, -1), 1))          # (k)
  fl    <- flow_diagonals(theta)

  PU <- Ps %*% U1                                   # (p x k)
  G  <- PU %*% diag(fl$lam1, k, k) %*% t(PU)

  if (ncol(Rs) > 0) {
    # Columns of B = t(Rs) %*% Pt %*% V1 are (-sin theta_j) * u2_j, matched to
    # angle j; normalize per column to recover the complement directions. The
    # complement rotation MUST share the same right basis V1 as U1 so the angles
    # pair column-for-column with PU.
    B     <- crossprod(Rs, Pt %*% V1)               # (p - k x k)
    norms <- sqrt(colSums(B^2))
    U2    <- -sweep(B, 2, ifelse(norms > 1e-12, norms, 1), "/")
    RU    <- Rs %*% U2                              # (p x k), paired with PU
    L2    <- diag(fl$lam2, k, k)
    L3    <- diag(fl$lam3, k, k)
    G <- G + PU %*% L2 %*% t(RU) + RU %*% L2 %*% t(PU) + RU %*% L3 %*% t(RU)
  }

  # Symmetric PSD square root: features are projected by G^(1/2) so that
  # Euclidean inner products in the transformed space equal the GFK kernel.
  G      <- 0.5 * (G + t(G))
  eig    <- eigen(G, symmetric = TRUE)
  G_half <- eig$vectors %*% diag(sqrt(pmax(eig$values, 0)), p, p) %*% t(eig$vectors)

  return(list(
    weighted_source_data = source_data %*% G_half,
    target_data          = target_data %*% G_half,
    G                    = G
  ))
}
