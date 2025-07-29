
#' Domain Adaptation via Aligned Riemannian Transport (ART)
#'
#' @description
#' The `domain_adaptation_art` function performs domain adaptation by aligning
#' the covariance structure of the source domain to that of the target domain
#' through Riemannian geometry. It utilizes the Ledoit–Wolf shrinkage estimator
#' for covariance estimation and applies a linear mapping derived from aligned
#' covariances under the Riemannian metric.
#'
#' @param source_data A numeric matrix representing the source domain (rows = observations,
#'   columns = features).
#' @param target_data A numeric matrix representing the target domain, with the same
#'   number of columns as \code{source_data}.
#'
#' @details
#' \strong{Aligned Riemannian Transport} proceeds as follows:
#' \enumerate{
#'   \item Center both source and target domains by subtracting their column means.
#'   \item Estimate the covariance matrices \eqn{C_S} and \eqn{C_T} using the Ledoit-Wolf
#'   shrinkage estimator.
#'   \item Align \eqn{C_S} to the geometry of \eqn{C_T} using Riemannian transport. If this fails,
#'   fall back to direct substitution with \eqn{C_T}.
#'   \item Compute a linear transformation \eqn{M = C_S^{-1/2} C_S^{aligned^{1/2}}} to transport
#'   source data into the target domain geometry.
#'   \item Apply this transformation to the centered source data and add back the target mean.
#' }
#'
#' @return A list containing:
#' \describe{
#'   \item{\code{weighted_source_data}}{Source data transformed to the target geometry.}
#'   \item{\code{target_data}}{Original (centered then restored) target data, unchanged.}
#'   \item{\code{transformation_matrix}}{The transformation matrix \eqn{M} used to align the domains.}
#' }
#'
#' @references
#' Zanini, P., Congedo, M., Jutten, C., Said, S., & Berthoumieu, Y. (2017)
#' \emph{Transfer learning: A Riemannian geometry framework with applications to
#' }brain–computer interfaces.*. IEEE Transactions on Biomedical Engineering, 65(5), 1107-1116.
#'
#' @examples
#' \dontrun{
#' set.seed(42)
#' src <- matrix(rnorm(100), nrow = 20)
#' tgt <- matrix(rnorm(100) + 1, nrow = 20)
#' out <- domain_adaptation_art(src, tgt)
#' }
#'
#' @export

# ==============================================================================
# Domain Adaptation with Aligned Riemannian Transport (ART)
# ==============================================================================
domain_adaptation_art <- function(source_data, target_data) {

  center <- function(X, mu) sweep(X, 2L, mu)               # fast column-centering
  ## Mean alignment
  mu_S <- colMeans(source_data)
  mu_T <- colMeans(target_data)
  Xs_c <- center(source_data, mu_S)
  Xt_c <- center(target_data, mu_T)

  ## Ledoit–Wolf shrinkage covariances (SPD)
  C_S <- LW_covariance(Xs_c)
  C_T <- LW_covariance(Xt_c)

  ## Aligned covariance (ART). If align_riemannian_transport is unstable,
  ## fall back to direct colouring with C_T (empirically robust).
  C_S_aligned <- tryCatch(
    align_riemannian_transport(list(C_S), list(C_T))[[1]],
    error = function(e) C_T
  )

  ## Linear map  M = C_S^{-½} · C_S_aligned^{½}
  C_S_inv_sqrt      <- matrix_power(C_S,        -0.5)
  C_S_aligned_sqrt  <- matrix_power(C_S_aligned, 0.5)
  M <- C_S_inv_sqrt %*% C_S_aligned_sqrt

  ## Transform, then restore the target mean
  Xs_aligned <- center(source_data, mu_S) %*% M
  Xs_aligned <- sweep(Xs_aligned, 2L, mu_T, '+')

  list(
    weighted_source_data = Xs_aligned,
    target_data          = target_data,
    transformation_matrix = M
  )
}

