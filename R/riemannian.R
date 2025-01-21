
#####################################
#' Riemannian-Distance-Based Alignment (RD)
#'
#' @description
#' The \code{domain_adaptation_riemannian} function measures the Riemannian distance
#' between two covariance matrices (\code{C_source} and \code{C_target}), then uses
#' a Procrustes-like alignment (via rotation) to map the source covariance into the
#' target domain. This approach is widely used in EEG or similar applications where
#' covariance-based alignment is beneficial to mitigate distribution shifts.
#'
#' @param C_source A numeric covariance matrix (source).
#' @param C_target A numeric covariance matrix (target), having the same dimensions
#'   as \code{C_source}.
#'
#' @return A list containing:
#' \describe{
#'   \item{\code{riemannian_distance}}{The Riemannian distance computed between
#'     \code{C_source} and \code{C_target}.}
#'   \item{\code{C_source_aligned}}{The source covariance matrix after alignment,
#'     approximating the target domain.}
#'   \item{\code{rotation_matrix}}{A rotation matrix that aligns \code{C_source}
#'     with \code{C_target} via singular value decomposition (SVD).}
#' }
#'
#' @details
#' \enumerate{
#'   \item \strong{Riemannian Distance} is calculated as \eqn{||\log(C_s^{-1/2} C_t C_s^{-1/2})||_F},
#'   reflecting how far two covariance matrices are on the manifold.
#'   \item \strong{Procrustes Alignment} uses SVD of both matrices (or one matrix
#'   inversion plus SVD) to find a rotation that aligns their principal axes.
#'   This step helps reduce distribution mismatch by making the source covariance
#'   resemble that of the target.
#' }
#'
#' @examples
#' \dontrun{
#' # Suppose we have two covariance matrices from EEG data
#' set.seed(123)
#' C_s <- crossprod(matrix(rnorm(50), 10, 5))
#' C_t <- crossprod(matrix(rnorm(50, mean = 2), 10, 5))
#'
#' rd_result <- domain_adaptation_riemannian(C_s, C_t)
#' cat("Riemannian distance =", rd_result$riemannian_distance, "\n")
#' str(rd_result$rotation_matrix)
#' }
#'
#' @export
#####################################

domain_adaptation_riemannian <- function(C_source, C_target) {
  # Step 1: Compute the Riemannian distance between source and target covariance matrices
  C1_inv <- solve(C_source)
  C_prod <- C1_inv %*% C_target
  eig_vals <- eigen(C_prod)$values
  log_eig_vals <- log(eig_vals)
  riemannian_distance <- sqrt(sum(log_eig_vals^2))

  # Step 2: Perform Procrustes adaptation (domain alignment)
  # Singular Value Decomposition (SVD) of source and target covariance matrices
  svd_source <- svd(C_source)
  svd_target <- svd(C_target)

  # Compute rotation matrix using U and V matrices from SVD
  rotation_matrix <- svd_source$u %*% t(svd_target$u)

  # Transform the source covariance matrix to align with the target domain
  C_source_aligned <- rotation_matrix %*% C_source %*% t(rotation_matrix)

  # Return both the Riemannian distance and the aligned covariance matrix
  return(list(riemannian_distance = riemannian_distance,
              C_source_aligned = C_source_aligned,
              rotation_matrix = rotation_matrix))
}
