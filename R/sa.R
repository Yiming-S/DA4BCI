
#####################################
#' Subspace Alignment (SA)
#'
#' @description
#' The `domain_adaptation_sa` function aligns principal component subspaces
#' between the source and target data. By performing separate PCA on each domain,
#' it obtains principal components for both, then learns a linear transformation
#' matrix that maps the source subspace to the target subspace, effectively reducing
#' distribution mismatch in a lower-dimensional feature space.
#'
#' @param source_data A numeric matrix representing the source domain (rows = observations,
#'   columns = features).
#' @param target_data A numeric matrix representing the target domain, with the same
#'   number of columns as \code{source_data}.
#' @param k An integer specifying the number of principal components to retain
#'   (default = 10). This value is capped by \code{min(ncol(source_data), ncol(target_data))}
#'   to ensure PCA dimensions are valid for both domains.
#'
#' @details
#' \strong{Subspace Alignment} proceeds as follows:
#' \enumerate{
#'   \item Perform PCA on \code{source_data} and \code{target_data} separately,
#'   each extracting the top \code{k} principal components, denoted \eqn{Z_s} and
#'   \eqn{Z_t}.
#'   \item Compute the alignment matrix \eqn{W = Z_s^T Z_t}, which linearly maps
#'   the source subspace onto the target subspace.
#'   \item Project the original source data (in PCA space) using \eqn{W} to
#'   obtain an aligned representation.
#'   \item The target data is already in its own PCA space, so it's returned
#'   for consistent comparison.
#' }
#'
#' @return A list containing:
#' \describe{
#'   \item{\code{weighted_source_data}}{The aligned source data, projected
#'     into the target's subspace.}
#'   \item{\code{target_data}}{The PCA-projected target data for consistent
#'     representation.}
#'   \item{\code{eigenvalue}}{The transformation matrix \eqn{W} used to align
#'     the source principal components with the target's.}
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' src <- matrix(rnorm(100), nrow = 20, ncol = 5)
#' tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)
#'
#' # Retain 3 principal components
#' sa_result <- domain_adaptation_sa(src, tgt, k = 3)
#' aligned_source <- sa_result$weighted_source_data
#' aligned_target <- sa_result$target_data
#'
#' dim(aligned_source)  # 20 x 3
#' dim(aligned_target)  # 20 x 3
#' }
#'
#' @export
#####################################

domain_adaptation_sa <- function(source_data, target_data, k = 10) { # , sigma = 1

  # Compute initial MMD
  # initial_mmd <- compute_mmd(source_data, target_data, sigma)
  # cat("Initial MMD:", initial_mmd, "\n")

  # Perform PCA and return the top h components based on the given k
  # perform_pca <- function(data, k) {
  # pca_result <- prcomp(data, scale. = TRUE)
  # h <- min(k, ncol(pca_result$x))
  # return(list(scores = pca_result$x[, 1:h], loadings = pca_result$rotation[, 1:h]))
  # }

  # Perform PCA on source and target data
  k <- min(k, ncol(source_data))
  # pca_source <- perform_pca(source_data, k)
  # pca_target <- perform_pca(target_data, k)
  # Z_s <- pca_source$scores
  # Z_t <- pca_target$scores
  pca_source <- prcomp(source_data, scale. = TRUE, rank. = k)
  pca_target <- prcomp(target_data, scale. = TRUE, rank. = k)
  Z_s <- pca_source$rotation
  Z_t <- pca_target$rotation

  # Ensure Z_s and Z_t have the same number of columns
  # h <- min(nrow(Z_s), nrow(Z_t))
  # Z_s <- Z_s[1:h, , drop = FALSE]
  # Z_t <- Z_t[1:h, , drop = FALSE]

  # Compute the linear transformation matrix W
  # W <- t(Z_s) %*% Z_t
  W <- crossprod(Z_s, Z_t)
  # W_source <- pca_source$loadings %*% W # not needed

  # Project the source and target data to the aligned subspaces
  # weighted_source_data <- source_data %*% W_source
  # projected_target_data <- target_data %*% pca_target$loadings
  weighted_source_data <- pca_source$x %*% W
  projected_target_data <- pca_target$x

  # final_mmd <- compute_mmd(weighted_source_data, projected_target_data, sigma)
  # cat("Final MMD:", final_mmd, "\n")

  # return(list(W_source = W_source, W_target = pca_target$loadings,
  # weighted_source_data = weighted_source_data,
  # target_data = projected_target_data))
  return(list(weighted_source_data = weighted_source_data,
              target_data = projected_target_data,
              eigenvalue = W))
}
