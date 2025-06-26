
#' Correlation Alignment (CORAL)
#'
#' @description
#' The `domain_adaptation_coral` function aligns the covariance structure of
#' source and target data by first "whitening" the source data, then "re-coloring"
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
#' # Make sure the `domain_adaptation_coral` function is available,
#' # e.g., library(DA4BCI) if it's in your DA4BCI package.
#'
#' # Set random seed
#' set.seed(123)
#' # Define parameters for generating source and target data
#' n_s <- n_t <- 10  # number of samples (in some abstract sense)
#' fs <- 160         # sampling frequency
#' t_seconds <- 3    # duration in seconds
#'
#' # Calculate adjusted sample sizes (rows of the data matrix)
#' adj_n_s <- n_s * fs * t_seconds
#' adj_n_t <- n_t * fs * t_seconds
#'
#' # Generate the source data (src) and target data (tgt)
#' # - Source data: random normal distribution
#' # - Target data: random normal distribution with a larger standard deviation (sd = 3)
#' src <- matrix(rnorm(adj_n_s * 50), adj_n_s, 50)
#' tgt <- matrix(rnorm(adj_n_t * 50, sd = 3), adj_n_t, 50)
#'
#' # Perform CORAL domain adaptation
#' da <- domain_adaptation_coral(src, tgt, lambda = 1e-5)
#' Z_s <- da$weighted_source_data  # aligned source data
#' Z_t <- da$target_data          # original target data (unmodified by CORAL)
#'
#' # Load visualization libraries
#' library(ggplot2)
#' library(gridExtra)
#'
#' # Compare distributions before and after alignment
#' plots <- plot_data_comparison(src, tgt, Z_s, Z_t,
#'                               description = "Normal")
#' combined_plot <- grid.arrange(plots$p1, plots$p2, ncol = 2,
#'                               top = paste("Method: CORAL",
#'                               "Normal Dist (Another Variant)"))
#' print(combined_plot)
#'
#' @export
domain_adaptation_coral <- function(source_data, target_data, lambda = 1e-5) {

  regularize_cov <- function(C, eps = 1e-6) {
  # Add a small jitter to the diagonal to ensure positive definiteness
  C + diag(eps * diag(C), nrow(C))}

  cov_source <- cov(source_data) + diag(lambda, ncol(source_data))
  cov_target <- cov(target_data) + diag(lambda, ncol(target_data))
  cov_source <- regularize_cov(cov_source, lambda)
  cov_target <- regularize_cov(cov_target, lambda)

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
