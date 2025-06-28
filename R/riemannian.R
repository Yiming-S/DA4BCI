

#' Riemannian-Distance-Based Alignment (RD)
#'
#' @description
#' The \code{domain_adaptation_riemannian} function computes the Riemannian distance
#' between two covariance matrices (\code{C_source} and \code{C_target}), then uses
#' an SVD-based Procrustes-like alignment (via a rotation matrix) to map the source
#' domain into the target domain. This approach is often applied in EEG or similar
#' scenarios where covariance-based alignment helps reduce distribution mismatches.
#'
#' @param source_data A numeric matrix representing the source domain, where rows
#'   correspond to observations and columns to features.
#' @param target_data A numeric matrix representing the target domain, with the
#'   same number of columns as \code{source_data}.
#'
#' @return A list containing:
#' \describe{
#'   \item{\code{weighted_source_data}}{The source data projected (rotated) to better
#'     match the target domain.}
#'   \item{\code{target_data}}{The original target data (unmodified).}
#'   \item{\code{rotation_matrix}}{A rotation matrix derived via SVD to align
#'     \code{C_source} with \code{C_target}.}
#'   \item{\code{cov_source_aligned}}{The covariance matrix of the source data after
#'     alignment (approximating the target domain).}
#' }
#'
#' @details
#' \enumerate{
#'   \item \strong{Riemannian Distance} is computed as \eqn{\sqrt{\sum (\log(\lambda_i))^2}},
#'     where \eqn{\lambda_i} are the eigenvalues of \eqn{C_s^{-1} C_t}. Note that this distance
#'     is currently computed but not returned.
#'   \item \strong{Procrustes Alignment} uses singular value decomposition (SVD) to find
#'     a rotation matrix that aligns the principal axes of \code{C_source} to
#'     those of \code{C_target}.
#' }
#'
#' @examples
#' # Make sure the `domain_adaptation_riemannian` function is available,
#' # e.g., library(DA4BCI) if it's in your DA4BCI package.
#'
#' # Set random seed
#' set.seed(123)
#' # Define parameters for generating source and target data
#' n_s <- n_t <- 10  # number of samples
#' fs <- 160         # sampling frequency
#' t_seconds <- 3    # duration in seconds
#'
#' # Calculate adjusted sample sizes (rows of the data matrix)
#' adj_n_s <- n_s * fs * t_seconds
#' adj_n_t <- n_t * fs * t_seconds
#'
#' # Generate source (src) and target (tgt) data
#' src <- matrix(rnorm(adj_n_s * 50), adj_n_s, 50)
#' tgt <- matrix(rnorm(adj_n_t * 50, sd = 3), adj_n_t, 50)
#'
#' # Perform domain adaptation
#' da <- domain_adaptation_riemannian(src, tgt)
#' Z_s <- da$weighted_source_data  # aligned source data
#' Z_t <- da$target_data           # original target data
#'
#' # Load visualization libraries
#' library(ggplot2)
#' library(gridExtra)
#'
#' # Compare distributions before and after alignment
#' plots <- plot_data_comparison(src, tgt, Z_s, Z_t,
#'                               description = "Normal")
#' combined_plot <- grid.arrange(plots$p1, plots$p2, ncol = 2,
#'                               top = paste("Method: riemannian",
#'                               "Normal Dist (Another Variant)"))
#' print(combined_plot)
#'
#' @export
domain_adaptation_riemannian <- function(source_data,
                                         target_data,
                                         ridge = 1e-6,
                                         pinv_tol = 1e-10) {


  ## 1. Covariance with ridge regularisation
  C_source <- cov(source_data) + diag(ridge, ncol(source_data))
  C_target <- cov(target_data) + diag(ridge, ncol(target_data))


  ## 2. Invert C_source
  inv_Cs <- tryCatch(
    solve(C_source),
    error = function(e) {
      sv <- svd(C_source)
      d_inv <- ifelse(sv$d > pinv_tol, 1 / sv$d, 0)
      sv$v %*% diag(d_inv) %*% t(sv$u)  # Moore-Penrose pinv
    }
  )


  ## 3. Riemannian distance
  eig_vals <- eigen(inv_Cs %*% C_target, symmetric = FALSE, only.values = TRUE)$values
  riem_dist <- sqrt(sum(log(Re(eig_vals))^2))  # purely real part is expected


  ## 4. Procrustes-like alignment (rotation)
  U_s <- svd(C_source)$u
  U_t <- svd(C_target)$u
  R   <- U_s %*% t(U_t)

  C_source_aligned <- R %*% C_source %*% t(R)


  ## 5. Return list
  list(
    weighted_source_data = source_data %*% R,
    target_data          = target_data,
    rotation_matrix      = R,
    cov_source_aligned   = C_source_aligned,
    riemannian_distance  = riem_dist
  )
}
