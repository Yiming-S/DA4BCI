
#####################################
#' Correlation Alignment (CORAL)
#'
#' @description
#' The `domain_adaptation_coral` function aligns the covariance structure of
#' source and target data by first “whitening” the source data, then “re-coloring”
#' it using the target covariance matrix. This unsupervised procedure helps reduce
#' domain shifts by matching second-order statistics (covariances) of the two datasets.
#'
#' @param source_data A numeric matrix representing the source domain, where rows
#'   correspond to observations and columns to features.
#' @param target_data A numeric matrix representing the target domain, with the
#'   same number of columns as \code{source_data}.
#' @param lambda A small positive scalar (default = 1e-5) added to the diagonal of
#'   both source and target covariance matrices to stabilize their inverses in
#'   high-dimensional or limited-sample cases.
#'
#' @return A list containing:
#' \describe{
#'   \item{\code{weighted_source_data}}{The transformed (whitened + recolored) source data
#'   so that its covariance now matches that of the target domain.}
#'   \item{\code{target_data}}{The original target data, returned for convenience.}
#' }
#'
#' @examples
#' # Example usage:
#' set.seed(123)
#' src <- matrix(rnorm(100), nrow = 20, ncol = 5)
#' tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)
#'
#' # Perform CORAL to align source data to target
#' res_coral <- domain_adaptation_coral(src, tgt, lambda = 1e-5)
#' aligned_src <- res_coral$weighted_source_data
#'
#' # Check covariances before/after
#' cat("Cov of target:\n")
#' print(cov(tgt))
#' cat("Cov of aligned source:\n")
#' print(cov(aligned_src))
#'
#' @export
#####################################

domain_adaptation_coral <- function(source_data, target_data, lambda = 1e-5) {
  cov_source <- cov(source_data) + diag(lambda, ncol(source_data))
  cov_target <- cov(target_data) + diag(lambda, ncol(target_data))

  # Whiten source data
  chol_decomp_s <- tryCatch(
    chol(solve(cov_source)),
    error = function(e) chol(MASS::ginv(cov_source))
  )

  # Color the whitened source data
  chol_decomp_t <- tryCatch(
    chol(cov_target),
    error = function(e) chol(MASS::ginv(cov_target))
  )

  source_data_aligned <- source_data %*% chol_decomp_s %*% chol_decomp_t

  list(
    weighted_source_data = source_data_aligned,
    target_data = target_data
  )
}
