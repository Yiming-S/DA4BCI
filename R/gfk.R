
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
#'   \item{\code{weighted_source_data}}{The source data projected by the GFK matrix \code{G}.}
#'   \item{\code{target_data}}{The target data projected by the same GFK matrix.}
#'   \item{\code{G}}{A \eqn{d \times d} geodesic flow kernel derived from the PCA bases
#'     of source and target.}
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

  # Step 1: PCA on source and target data
  dim_subspace <- min(dim_subspace, ncol(source_data), ncol(target_data))
  pca_s <- prcomp(source_data, scale. = TRUE, rank. = dim_subspace)
  pca_t <- prcomp(target_data, scale. = TRUE, rank. = dim_subspace)

  # Us and Ut are d x dim_subspace with orthonormal columns
  Us <- pca_s$rotation
  Ut <- pca_t$rotation

  d <- nrow(Us)
  k <- dim_subspace

  # Step 2: Orthonormal complements Qs, Qt
  Qs <- orthonormal_complement(Us)
  Qt <- orthonormal_complement(Ut)

  # Step 3: Construct the intermediate matrices for the geodesic
  U0 <- cbind(Us, Qs)
  U1 <- cbind(Ut, Qt)

  # Compute M0 = U0^T * U1 and do SVD on it
  M0 <- t(U0) %*% U1
  svd_M0 <- svd(M0)

  U_m <- svd_M0$u
  S_m <- svd_M0$d
  V_m <- svd_M0$v

  # Clamp singular values to [0,1] and compute principal angles
  S_clamped <- pmin(S_m, 1)
  angles <- acos(S_clamped)

  # Step 4: Build the GFK kernel G from the angles (simplified approach)
  int_cossin <- function(ti) {
    if (ti < 1e-12) {
      # Small-angle approximation
      return(c(1, 0))
    } else {
      e_val <- (ti - sin(ti)) / (ti)
      f_val <- (1 - cos(ti)) / (ti)
      return(c(e_val, f_val))
    }
  }

  # Initialize a diagonal vector for integrating the angles
  g_vec <- rep(1, d)
  for (i in seq_len(k)) {
    tmp <- int_cossin(angles[i])
    e_i <- tmp[1]
    g_vec[i] <- e_i
  }
  diag_g <- diag(g_vec, d, d)

  # Construct G = U0 * U_m * diag_g * U_m^T * U0^T
  G <- U0 %*% U_m %*% diag_g %*% t(U_m) %*% t(U0)

  # Step 5: Map source and target data using G
  source_data_gfk <- source_data %*% G
  target_data_gfk <- target_data %*% G

  return(list(
    weighted_source_data = source_data_gfk,
    target_data = target_data_gfk,
    G = G
  ))
}
