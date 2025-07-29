
#' Manifold-based Multi-step Domain Adaptation (M3D)
#'
#' @description
#' The `domain_adaptation_m3d` function performs two-stage manifold-based domain
#' adaptation using a dynamic distribution alignment strategy. It first reduces
#' domain discrepancy via kernel-based or subspace-based alignment (e.g., TCA, SA),
#' followed by iterative refinement of class-aware alignment with pseudo-labels.
#'
#' @param source_data A numeric matrix representing the source domain (rows = samples,
#'   columns = features).
#' @param source_labels A factor or vector of class labels for the source domain.
#' @param target_data A numeric matrix representing the target domain with the same
#'   number of columns as \code{source_data}.
#' @param stage1 A list specifying the method and control parameters for the first
#'   alignment stage (e.g., TCA, PCA). Default: \code{list(method = "tca", control = list(k = NULL, sigma = 1))}.
#' @param stage2 A list specifying the method and control parameters for the second
#'   alignment stage (e.g., SA). Default: \code{list(method = "sa", control = list(k = 10))}.
#' @param l_iter Number of iterations for label refinement and class-wise alignment. Default: 10.
#' @param lambda_ridge Regularization strength for the ridge regression step. Default: 1e-2.
#' @param eta_kernel Weight for class-aware kernel alignment term. Default: 0.1.
#' @param label_offset Optional offset to shift the pseudo-labels. Default: 0.
#' @param expl_var Explained variance threshold (used in PCA-based dimension selection). Default: 0.90.
#' @param max_dim Maximum cap for automatically selected subspace dimension. Default: 30.
#'
#' @details
#' \strong{Multi-step Manifold Alignment with Dynamic Distribution (M3D)} proceeds as:
#' \enumerate{
#'   \item Use stage-1 transformation (e.g., TCA) to reduce marginal domain shift.
#'   \item Apply optional stage-2 transformation (e.g., SA) to further align subspaces.
#'   \item Construct a kernel matrix over aligned features and apply a class-aware MMD
#'         penalty using pseudo-labels for the target domain.
#'   \item Iteratively update pseudo-labels via ridge regression and recompute
#'         dynamic alignment matrices.
#'   \item Perform eigen-decomposition on the final kernel matrix to extract features
#'         in the aligned manifold space.
#' }
#'
#' This implementation is based on the M3D framework proposed in:
#' Luo, T. et al. (2024). M3D: Manifold-based domain adaptation with dynamic distribution for
#' non-deep transfer learning in cross-subject and cross-session EEG-based emotion recognition.
#' \emph{arXiv preprint arXiv:2404.15615}.
#'
#' @return A list with:
#' \describe{
#'   \item{\code{weighted_source_data}}{The aligned source features after two-stage transformation and M3D refinement.}
#'   \item{\code{target_data}}{The aligned target features in the same latent manifold space.}
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' src <- matrix(rnorm(100 * 20), nrow = 100)
#' tgt <- matrix(rnorm(100 * 20, mean = 1), nrow = 100)
#' labels <- sample(1:3, 100, replace = TRUE)
#' out <- domain_adaptation_m3d(src, labels, tgt)
#' str(out$weighted_source_data)
#' str(out$target_data)
#' }
#'
#' @references
#' Luo, T., Zhang, J., Qiu, Y., Zhang, L., Hu, Y., Yu, Z., & Liang, Z. (2024).
#' M3D: Manifold-based domain adaptation with dynamic distribution for non-deep transfer learning
#' in cross-subject and cross-session EEG-based emotion recognition.
#' \emph{arXiv preprint} arXiv:2404.15615.
#'
#' @export
#'
domain_adaptation_m3d <- function(
    source_data, source_labels, target_data,
    stage1 = list(method = "tca", control = list(k = NULL, sigma = 1)),
    stage2 = list(method = "sa", control = list(k = 10)),
    l_iter = 10, lambda_ridge = 1e-2, eta_kernel = 0.1, label_offset = 0,
    expl_var = 0.90, max_dim = 30)
{

  to_onehot <- function(y) {
    y <- factor(y); Y <- model.matrix(~ y - 1)
    colnames(Y) <- levels(y); list(Y = Y, levels = levels(y))
  }
  center_kernel <- function(K) {
    n <- nrow(K); H <- diag(n) - 1/n; H %*% K %*% H
  }
  auto_dim <- function(X, keep = expl_var, cap = max_dim) {
    # Using prcomp for better stability
    pca <- prcomp(X, scale. = TRUE, center = TRUE)
    cum_var <- cumsum(pca$sdev^2) / sum(pca$sdev^2)
    pmin(which(cum_var >= keep)[1], cap, na.rm = TRUE)
  }

  # Auto-dimensioning
  if (is.null(stage1$control$k))
    stage1$control$k <- auto_dim(rbind(source_data, target_data))
  if (is.null(stage2$control$dim_subspace))
    stage2$control$dim_subspace <- min(stage1$control$k, max_dim)

  # Stage-1 / Stage-2: Initial Transformation
  # Assumes an external `domain_adaptation` function is available
  da1 <- domain_adaptation(source_data, target_data,
                           method = stage1$method, control = stage1$control)
  Zs1 <- da1$weighted_source_data;  Zt1 <- da1$target_data

  if (!is.null(stage2)) {
    da2 <- domain_adaptation(Zs1, Zt1,
                             method = stage2$method, control = stage2$control)
    Zs <- da2$weighted_source_data; Zt <- da2$target_data
    K_all <- if (!is.null(da2$K)) da2$K else tcrossprod(rbind(Zs, Zt))
  } else {
    Zs <- Zs1; Zt <- Zt1;  K_all <- tcrossprod(rbind(Zs, Zt))
  }
  K_all <- center_kernel(K_all)

  n_s <- nrow(Zs); n_t <- nrow(Zt); n_all <- n_s + n_t
  idx_s <- 1:n_s; idx_t <- (n_s + 1):n_all
  oh <- to_onehot(source_labels); Ysrc <- oh$Y
  Kss <- K_all[idx_s, idx_s]; Kts <- K_all[idx_t, idx_s]

  # Initial pseudo-labels
  W <- solve(Kss + lambda_ridge * diag(n_s), Ysrc)
  Y_t <- apply(Kts %*% W, 1, which.max)

  # Iterative alignment to find the best M matrix
  Z_all <- rbind(Zs, Zt)
  M <- matrix(0, n_all, n_all) # Initialize M
  for (it in seq_len(l_iter)) {
    M0 <- matrix(-1/(n_s*n_t), n_all, n_all)
    M0[idx_s, idx_s] <- 1/n_s^2; M0[idx_t, idx_t] <- 1/n_t^2

    Mc <- matrix(0, n_all, n_all)
    for (c in seq_len(ncol(Ysrc))) {
      S <- idx_s[ Ysrc[,c] == 1 ]
      T <- idx_t[ Y_t      == c ]
      if (!length(S) || !length(T)) next
      Mc[S,S] <- Mc[S,S] + 1/length(S)^2
      Mc[T,T] <- Mc[T,T] + 1/length(T)^2
      Mc[S,T] <- Mc[S,T] - 1/(length(S)*length(T))
      Mc[T,S] <- Mc[T,S] - 1/(length(S)*length(T))
    }

    tr_Mc <- sum(Z_all * (Mc %*% Z_all))
    tr_tot <- tr_Mc + sum(Z_all * (M0 %*% Z_all))
    mu <- if (tr_tot < 1e-12) 0 else tr_Mc / tr_tot
    M <- (1 - mu) * M0 + mu * Mc

    Hs <- Kss + eta_kernel * M[idx_s, idx_s]
    W <- solve(Hs + lambda_ridge * diag(n_s), Ysrc - label_offset)
    Y_t <- apply(Kts %*% W, 1, which.max)
  }


  # 1. Construct the final, fully aligned kernel matrix
  K_aligned <- K_all + eta_kernel * M

  # 2. Perform eigendecomposition (like Kernel PCA)
  # This finds the principal components in the aligned kernel space.
  eig_decomp <- eigen(K_aligned, symmetric = TRUE)

  # 3. Create the transformed features
  # The new features are the projections onto the eigenvectors,
  # scaled by the sqrt of eigenvalues.
  # We only keep components with positive eigenvalues.
  k <- stage2$control$dim_subspace # Reuse the dimension parameter

  eigenvalues <- eig_decomp$values[1:k]
  eigenvectors <- eig_decomp$vectors[, 1:k]

  # Handle potential negative eigenvalues due to numerical precision
  eigenvalues[eigenvalues < 0] <- 0

  # The new feature representation for ALL data
  Z_all_transformed <- eigenvectors %*% diag(sqrt(eigenvalues))

  # 4. Split back into source and target
  Zs_transformed <- Z_all_transformed[idx_s, ]
  Zt_transformed <- Z_all_transformed[idx_t, ]

  # Return the NEW, TRANSFORMED features
  list(
    weighted_source_data = Zs_transformed,
    target_data = Zt_transformed
  )
}
