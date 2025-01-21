#' Transfer Component Analysis (TCA) for Domain Adaptation
#'
#' @description
#' This function implements TCA to align source and target data distributions,
#' reducing cross-domain mismatch.
#'
#' @param source_data A numeric matrix (observations by rows, features by columns)
#'   representing the source domain.
#' @param target_data A numeric matrix of the same shape for the target domain.
#' @param k An integer specifying the number of latent components. Default = 10.
#' @param sigma A numeric value for the RBF kernel bandwidth. Default = 1.
#' @param mu A numeric regularization term. Default = 1.
#'
#' @return A list containing:
#' \describe{
#'   \item{weighted_source_data}{Transformed source data in TCA subspace.}
#'   \item{target_data}{Transformed target data in TCA subspace.}
#'   \item{eigenvalue}{The matrix of top k eigenvectors used for projection.}
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' src <- matrix(rnorm(100), nrow=20, ncol=5)
#' tgt <- matrix(rnorm(100, mean=2), nrow=20, ncol=5)
#' res <- domain_adaptation_tca(src, tgt, k=3, sigma=1, mu=0.1)
#' str(res)
#' }
#'
#' @export

domain_adaptation_tca <- function(source_data, target_data,
                                  k = 10, sigma = 1, mu = 1) {

  # Compute initial MMD
  # initial_mmd <- compute_mmd(source_data, target_data, sigma)
  # cat("Initial MMD:", initial_mmd, "\n")

  # Combine source and target data
  X <- rbind(source_data, target_data)

  # Compute the kernel matrix using RBF kernel
  # K <- rbf_kernel(X, X, sigma)
  nrmX <- rowSums(X^2)
  K <- exp( (-0.5/sigma^2) * (outer(nrmX, nrmX, "+") - 2 * tcrossprod(X)) ) # faster


  # Construct the L matrix
  n_s <- nrow(source_data)
  n_t <- nrow(target_data)
  # L <- construct_L_matrix(n_s, n_t) # not needed

  # Compute the centering matrix H
  n <- n_s + n_t
  # H <- diag(n) - (1 / n) * matrix(1, n, n) # not needed

  # Solve the generalized eigenvalue problem
  # KLK <- K %*% L %*% K # not needed
  # KHK <- K %*% H %*% K
  KHK <- tcrossprod(K - rowMeans(K)) # much faster
  # M <- KLK + mu * diag(n) # not needed

  # eigen_result <- tryCatch(geigen(M, KH, TRUE),
  # error = function(e) geigen(M, KH, FALSE))
  # eigen_result <- geigen(M, KHK, FALSE) # KHK is not psd since H is not full rank
  # KLK = aa' (note that L is rank-1)
  a <- rowMeans(K[,1:n_s]) - rowMeans(K[,-(1:n_s)])
  # A = mu I + KLK = mu I + aa'
  # A^{-1} = (1/mu) I - (aa') / (mu * (mu + a'a))
  cst <- mu * (mu + sum(a^2))
  # B = KHK
  B <- KHK
  invAB <- B / mu - (a / cst) %*% crossprod(a, B)
  W <- eigs(invAB, k)$vectors
  if (is.complex(W)) W <- Re(W)

  # For testing inversion formula
  # A <- diag(mu, n) + tcrossprod(a)
  # invAB <- solve(A, B)
  # test <- A %*% invAB - B
  # range(test)

  # Rescale solutions to have norm 1 in metric B
  nrm <- colSums(W * (B %*% W))
  if (any(nrm <= 0)) {
    W <- W[, nrm > 0]
    nrm <- nrm[nrm > 0]
  }
  W <- sweep(W, 2, sqrt(nrm), "/")

  # # Select the top k eigenvectors
  # W <- eigen_result$vectors[, 1:k]
  # NO! Here you should take the eigenvectors associated
  # to the k *smallest* generalized eigenvalues

  # Project the data
  Z <- K %*% W

  # Split the projected data back into source and target
  weighted_source_data <- Z[1:n_s, ]
  projected_target_data <- Z[-(1:n_s), ]

  # final_mmd <- compute_mmd(weighted_source_data, projected_target_data, sigma)
  # cat("Final MMD:", final_mmd, "\n")

  return(list(weighted_source_data = weighted_source_data,
              target_data = projected_target_data,
              eigenvalue = W))
}
