
#####################################
#' Maximum/Minimum Independence Domain Adaptation (MIDA)
#'
#' @description
#' The `domain_adaptation_mida` function augments source and target data with
#' domain labels (0 for source, 1 for target) and applies a kernel-based approach
#' (via HSIC) to either \emph{maximize} or \emph{minimize} the dependence between
#' the feature space and these domain labels. By setting \code{max = TRUE} or
#' \code{max = FALSE}, one can highlight or suppress domain-specific differences,
#' helping to better align source and target distributions for improved transfer.
#'
#' @param source_data A numeric matrix representing the source domain,
#'   with rows as observations and columns as features.
#' @param target_data A numeric matrix representing the target domain,
#'   with the same number of columns as \code{source_data}.
#' @param k An integer specifying the number of projected components
#'   to retain in the final subspace. Default is 10.
#' @param sigma A numeric bandwidth parameter for the RBF kernel used
#'   in the domain-adaptation procedure. Default is 1.
#' @param mu A numeric regularization term (default = 0.1) that balances
#'   the independence objective against potential overfitting.
#' @param max A logical indicating whether to maximize (\code{TRUE}) or minimize
#'   (\code{FALSE}) the domain dependence. Default is \code{TRUE}.
#'
#' @return A list containing:
#' \describe{
#'   \item{\code{weighted_source_data}}{Source data transformed into the MIDA subspace.}
#'   \item{\code{target_data}}{Target data transformed into the same MIDA subspace.}
#'   \item{\code{eigenvalue}}{The matrix of top \code{k} eigenvectors used for the projection.}
#' }
#'
#' @details
#' \strong{MIDA (Maximum/Minimum Independence Domain Adaptation)} constructs
#' an RBF kernel for both feature data and domain labels, then formulates a
#' generalized eigenvalue problem under the HSIC framework. When \code{max = TRUE},
#' it seeks to maximize the association between features and the domain label,
#' potentially accentuating domain-specific structures. When \code{max = FALSE},
#' it aims to \emph{minimize} this association to reduce cross-domain discrepancy.
#' The parameter \code{mu} adds a regularization term that prevents overfitting
#' and stabilizes the solution in high-dimensional settings.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' src <- matrix(rnorm(100), nrow = 20, ncol = 5)
#' tgt <- matrix(rnorm(100, mean = 3), nrow = 20, ncol = 5)
#'
#' # Maximize domain dependence
#' res_mida_max <- domain_adaptation_mida(src, tgt, k = 3, sigma = 1, mu = 0.1, max = TRUE)
#' aligned_src_max <- res_mida_max$weighted_source_data
#'
#' # Minimize domain dependence
#' res_mida_min <- domain_adaptation_mida(src, tgt, k = 3, sigma = 1, mu = 0.1, max = FALSE)
#' aligned_src_min <- res_mida_min$weighted_source_data
#' }
#'
#' @export
#####################################

domain_adaptation_mida <- function(source_data, target_data,
                                   k = 10, sigma = 1, mu = 0.1,
                                   max = TRUE) {
  n_s <- nrow(source_data)
  n_t <- nrow(target_data)

  # Create domain feature vectors
  source_domain_labels <- rep(0, n_s)
  target_domain_labels <- rep(1, n_t)

  # Combine source and target data
  data_combined <- rbind(source_data, target_data)
  domain_features <- c(source_domain_labels, target_domain_labels)

  # Augment data with domain labels
  augmented_data <- cbind(data_combined, domain_features)

  # Compute the kernel matrix using RBF kernel
  nrm_augmented <- rowSums(augmented_data^2)
  K <- exp((-0.5 / sigma^2) * (outer(nrm_augmented, nrm_augmented, "+") -
                                 2 * tcrossprod(augmented_data)))

  # Construct the kernel matrices for HSIC
  K_x <- K
  K_d <- outer(domain_features, domain_features, FUN = function(x, y) as.numeric(x == y))

  # Centering matrix
  n <- n_s + n_t
  H <- diag(n) - (1 / n) * matrix(1, n, n)

  # Build the M matrix (maximum or minimum domain dependence)
  if (max) {
    M <- K_x %*% H %*% K_d %*% H %*% K_x + mu * K_x %*% H %*% K_x
  } else {
    M <- K_x %*% H %*% K_d %*% H %*% K_x - mu * K_x %*% H %*% K_x
  }

  # Compute the KH matrix
  KH <- K_x %*% H %*% K_x

  # Solve the generalized eigenvalue problem
  eigen_result <- tryCatch(
    geigen(M, KH, TRUE),
    error = function(e) geigen(M, KH, FALSE)
  )

  # Select and process the top k eigenvectors
  W <- eigen_result$vectors[, 1:k]
  W <- Re(W)

  # Project the data
  Z <- K_x %*% W
  Z <- Re(Z)

  # Split the projected data back into source and target
  weighted_source_data <- Z[1:n_s, ]
  projected_target_data <- Z[-(1:n_s), ]

  return(list(weighted_source_data = weighted_source_data,
              target_data = projected_target_data,
              eigenvalue = W))
}
