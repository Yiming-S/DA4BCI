
# rbf_kernel
# compute_distance_matrix
# compute_wasserstein
# compute_mmd
# orthonormal_complement
# evaluate_shift
# plot_data_comparison

#####################################
#' RBF (Gaussian) Kernel Computation
#'
#' @description
#' The \code{rbf_kernel} function calculates the Radial Basis Function (RBF)
#' kernel matrix between two sets of observations \code{x} and \code{y},
#' using \eqn{\exp(-||x - y||^2 / \sigma^2)}. It returns only the cross-block
#' corresponding to \code{x} vs. \code{y}.
#'
#' @param x A numeric matrix where rows correspond to observations and columns
#'   to features.
#' @param y A numeric matrix with the same number of columns as \code{x}.
#' @param sigma A positive scalar for the RBF kernel bandwidth.
#'
#' @details
#' This function computes pairwise distances among rows of \code{x} and \code{y}
#' (stacked together) via \code{\link[stats]{dist}}, then transforms them using
#' the Gaussian kernel formula. Only the \code{x} vs. \code{y} sub-block is returned,
#' producing an \eqn{n_x \times n_y} matrix.
#'
#' @return A numeric matrix of size \eqn{nrow(x) \times nrow(y)} containing
#'   the RBF kernel values.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' x <- matrix(rnorm(20), nrow = 5, ncol = 4)
#' y <- matrix(rnorm(24, mean = 2), nrow = 6, ncol = 4)
#' Kxy <- rbf_kernel(x, y, sigma = 1)
#' dim(Kxy)  # 5 x 6
#' }
#'
#' @export
#####################################
rbf_kernel <- function(x, y, sigma) {
  distx <- as.matrix(dist(rbind(x, y)))
  kernel <- exp(- (distx / sigma)^2)
  n <- nrow(x)
  return(kernel[1:n, -(1:n)])
}

####################################
#' Compute Euclidean Distance Matrix
#'
#' @description
#' The \code{compute_distance_matrix} function computes the pairwise Euclidean
#' distance matrix between two datasets \code{source} and \code{target}.
#'
#' @param source A numeric matrix where rows are observations and columns are features.
#' @param target A numeric matrix with the same structure as \code{source}.
#'
#' @return A numeric matrix of size \eqn{nrow(source) \times nrow(target)} containing
#'   the pairwise Euclidean distances.
#'
#' @examples
#' \dontrun{
#' source <- matrix(rnorm(20), nrow = 5, ncol = 4)
#' target <- matrix(rnorm(24, mean = 2), nrow = 6, ncol = 4)
#' dist_matrix <- compute_distance_matrix(source, target)
#' dim(dist_matrix)  # 5 x 6
#' }
#'
#' @export
####################################
compute_distance_matrix <- function(source, target) {
  cross_term <- source %*% t(target)
  source_norms <- rowSums(source^2)
  target_norms <- rowSums(target^2)
  sqrt(outer(source_norms, target_norms, "+") - 2 * cross_term)
}

####################################
#' Compute Wasserstein Distance
#'
#' @description
#' The \code{compute_wasserstein} function calculates the Wasserstein distance
#' between the distributions represented by \code{source} and \code{target}.
#'
#' @param source A numeric matrix where rows are observations and columns are features.
#' @param target A numeric matrix with the same structure as \code{source}.
#'
#' @return A single numeric value representing the Wasserstein distance.
#'
#' @details
#' This function uses the \code{transport} package to compute the optimal transport plan
#' and calculate the Wasserstein distance. Pairwise Euclidean distances are used as the cost matrix.
#'
#' @examples
#' \dontrun{
#' source <- matrix(rnorm(20), nrow = 5, ncol = 4)
#' target <- matrix(rnorm(24, mean = 2), nrow = 6, ncol = 4)
#' wasserstein_dist <- compute_wasserstein(source, target)
#' cat("Wasserstein Distance:", wasserstein_dist, "\n")
#' }
#'
#' @export
####################################
compute_wasserstein <- function(source, target) {
  stopifnot(is.matrix(source), is.matrix(target))
  if (ncol(source) != ncol(target)) {
    stop("Source and target must have the same number of features (columns).")
  }

  n_s <- nrow(source)
  n_t <- nrow(target)
  p <- rep(1 / n_s, n_s)
  q <- rep(1 / n_t, n_t)

  cost_matrix <- compute_distance_matrix(source, target)
  plan <- transport::transport(p, q, cost_matrix, method = "revsimplex")
  sum(plan$mass * cost_matrix[cbind(plan$from, plan$to)])
}

#####################################
#' Compute the Maximum Mean Discrepancy (MMD)
#'
#' @description
#' The \code{compute_mmd} function estimates the squared MMD between a source
#' dataset and a target dataset, using an RBF kernel with bandwidth \code{sigma}.
#' MMD is often used to measure distribution discrepancy in domain adaptation.
#'
#' @param source A numeric matrix representing the source domain (rows = observations,
#'   columns = features).
#' @param target A numeric matrix representing the target domain, with the same
#'   number of columns as \code{source}.
#' @param sigma A positive scalar for the RBF kernel bandwidth.
#'
#' @details
#' The MMD is computed as:
#' \deqn{
#'   \mathrm{MMD}^2 = \frac{1}{m^2} \sum_{i,j} K_{ss}(i,j)
#'     + \frac{1}{n^2} \sum_{i,j} K_{tt}(i,j)
#'     - \frac{2}{mn} \sum_{i,j} K_{st}(i,j),
#' }
#' where \eqn{K_{ss}} is the RBF kernel among source samples, \eqn{K_{tt}} among
#' target samples, and \eqn{K_{st}} between source and target samples. This metric
#' reflects how different the distributions are.
#'
#' @return A single numeric value indicating the estimated squared MMD between
#'   \code{source} and \code{target}.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' src <- matrix(rnorm(100), nrow = 20, ncol = 5)
#' tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)
#' mmd_val <- compute_mmd(src, tgt, sigma = 1)
#' cat("MMD^2 =", mmd_val, "\n")
#' }
#'
#' @export
#####################################
compute_mmd <- function(source, target, sigma) {
  Kss <- rbf_kernel(source, source, sigma)
  Ktt <- rbf_kernel(target, target, sigma)
  Kst <- rbf_kernel(source, target, sigma)

  m <- nrow(source)
  n <- nrow(target)

  mmd <- sum(Kss) / (m * m) + sum(Ktt) / (n * n) - 2 * sum(Kst) / (m * n)
  return(mmd)
}


#####################################
#' Orthonormal Complement of a Matrix
#'
#' @description
#' The \code{orthonormal_complement} function computes a matrix whose columns form
#' an orthonormal basis for the subspace orthogonal to \code{U} in \eqn{\mathbb{R}^d}.
#' Useful in methods like GFK, where one needs both the principal subspace and
#' its complement.
#'
#' @param U A \eqn{d \times k} matrix with orthonormal columns.
#'
#' @details
#' The function uses a QR decomposition of the \eqn{d \times d} identity matrix
#' to get a full orthonormal basis, then projects it onto the orthogonal complement
#' of \code{U}. Columns close to zero (norm < 1e-8) are removed. The result is truncated
#' to \eqn{d-k} columns if needed.
#'
#' @return A \eqn{d \times (d-k)} matrix (or fewer columns if some are removed),
#'   with orthonormal columns that are orthogonal to every column of \code{U}.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' U <- qr.Q(qr(matrix(rnorm(12), nrow = 4, ncol = 3)))  # 4x3 orthonormal basis
#' U_perp <- orthonormal_complement(U)
#' dim(U_perp)  # should be 4 x 1
#' # Check orthogonality: crossprod(U, U_perp) ~ 0
#' }
#'
#' @export
#####################################
orthonormal_complement <- function(U) {
  d <- nrow(U)
  k <- ncol(U)
  Q <- qr.Q(qr(diag(d)))
  ProjU <- U %*% t(U)

  for (i in seq_len(d)) {
    qi <- Q[, i, drop = FALSE]
    qi_orth <- qi - ProjU %*% qi
    norm_qi <- sqrt(sum(qi_orth^2))
    if (norm_qi > 1e-12) {
      Q[, i] <- qi_orth[, 1] / norm_qi
    } else {
      Q[, i] <- 0
    }
  }

  norms <- colSums(Q^2)
  keep_idx <- which(norms > 1e-8)
  U_perp <- Q[, keep_idx, drop = FALSE]

  if (ncol(U_perp) > (d - k)) {
    U_perp <- U_perp[, 1:(d - k), drop = FALSE]
  }

  return(U_perp)
}

####################################
#' Evaluate Distribution Shift
#'
#' @description
#' The \code{evaluate_shift} function computes two metrics (MMD and Wasserstein distance)
#' to evaluate the distributional shift between \code{source_data} and \code{target_data},
#' both before and after domain adaptation.
#'
#' @param source_data A numeric matrix representing the source domain data.
#' @param target_data A numeric matrix representing the target domain data.
#' @param adapted_source A numeric matrix representing the adapted source domain data.
#' @param adapted_target A numeric matrix representing the adapted target domain data.
#'
#' @return A data frame with metrics (MMD, Wasserstein) and their values before
#'   and after adaptation.
#'
#' @examples
#' \dontrun{
#' source <- matrix(rnorm(200), nrow = 20, ncol = 10)
#' target <- matrix(rnorm(200, mean = 2), nrow = 20, ncol = 10)
#' adapted_source <- matrix(rnorm(200), nrow = 20, ncol = 10)
#' adapted_target <- matrix(rnorm(200, mean = 1.5), nrow = 20, ncol = 10)
#'
#' results <- evaluate_shift(source, target, adapted_source, adapted_target)
#' print(results)
#' }
#'
#' @export
####################################
evaluate_shift <- function(
    source_data, target_data,
    adapted_source, adapted_target
) {
  mmd_before <- compute_mmd(source_data, target_data, sigma = 1)
  mmd_after <- compute_mmd(adapted_source, adapted_target, sigma = 1)

  wasserstein_before <- compute_wasserstein(source_data, target_data)
  wasserstein_after <- compute_wasserstein(adapted_source, adapted_target)

  data.frame(
    Metric = c("MMD", "Wasserstein"),
    Before = c(mmd_before, wasserstein_before),
    After = c(mmd_after, wasserstein_after)
  )
}

#####################################
#' Visual Comparison of Data Before and After Domain Adaptation
#'
#' @description
#' The \code{plot_data_comparison} function provides a simple 2D visualization
#' comparing source and target data distributions \emph{before} and \emph{after}
#' a domain adaptation transform. It uses either PCA or t-SNE to reduce data to
#' two dimensions for plotting.
#'
#' @param source_data A numeric matrix of the source data (rows = observations),
#'   before adaptation.
#' @param target_data A numeric matrix of the target data, before adaptation,
#'   with the same number of columns as \code{source_data}.
#' @param Z_s (Optional) The source data after adaptation; if \code{NULL}, no
#'   "after" plot is generated.
#' @param Z_t (Optional) The target data after adaptation.
#' @param description A character string to annotate the plot titles.
#' @param method A character string indicating which dimensional reduction method
#'   to apply for visualization. Choices are \code{"pca"} (default) or \code{"tsne"}.
#'
#' @details
#' \enumerate{
#'   \item Merges \code{source_data} and \code{target_data}, applies either PCA
#'     (\code{\link[stats]{prcomp}}) or t-SNE (\code{\link[Rtsne]{Rtsne}}) to map
#'     them into 2D, then plots them as "Before."
#'   \item If \code{Z_s} and \code{Z_t} are provided, merges and plots them as "After"
#'     the adaptation transform.
#' }
#' This helps visually inspect how the adaptation influences the overlap or
#' separation of the source and target.
#'
#' @return A list containing:
#' \describe{
#'   \item{\code{p1}}{The \code{ggplot2} object for the "before" distribution.}
#'   \item{\code{p2}}{The \code{ggplot2} object for the "after" distribution
#'     (only returned if \code{Z_s} and \code{Z_t} are not \code{NULL}).}
#' }
#'
#' @examples
#' \dontrun{
#' library(ggplot2)
#' set.seed(123)
#' src <- matrix(rnorm(100), nrow=20, ncol=5)
#' tgt <- matrix(rnorm(100, mean=3), nrow=20, ncol=5)
#'
#' # Plot only "before"
#' p_list <- plot_data_comparison(src, tgt, description = "NoAdapt", method = "pca")
#' print(p_list$p1)
#'
#' # Suppose Z_s and Z_t are aligned data
#' Z_s <- src + 1
#' Z_t <- tgt
#' p_list2 <- plot_data_comparison(src, tgt, Z_s, Z_t,
#'                                 description = "FakeAlign", method = "tsne")
#' print(p_list2$p1)
#' print(p_list2$p2)
#' }
#'
#' @export
#####################################
plot_data_comparison <- function(source_data, target_data,
                                 Z_s = NULL, Z_t = NULL,
                                 description = "NULL",
                                 method = c("pca", "tsne")) {
  method <- match.arg(method)

  to_real <- function(data) {
    if (is.complex(data)) Re(data) else data
  }

  source_data <- to_real(source_data)
  target_data <- to_real(target_data)
  if (!is.null(Z_s)) Z_s <- to_real(Z_s)
  if (!is.null(Z_t)) Z_t <- to_real(Z_t)

  data_before <- rbind(source_data, target_data)
  labels_before <- c(rep("Source", nrow(source_data)),
                     rep("Target", nrow(target_data)))

  pca_before <- switch(
    method,
    pca  = prcomp(data_before, rank. = 2)$x,
    tsne = Rtsne::Rtsne(data_before)$Y
  )
  df_before <- data.frame(PC1 = pca_before[,1],
                          PC2 = pca_before[,2],
                          Label = labels_before)

  p1 <- ggplot2::ggplot(df_before, ggplot2::aes(x = PC1, y = PC2, color = Label)) +
    ggplot2::geom_point(alpha = 0.6) +
    ggplot2::labs(title = paste0("Data Distribution Before ", description))

  if (is.null(Z_s) || is.null(Z_t)) {
    return(list(p1 = p1))
  }

  data_after <- rbind(Z_s, Z_t)
  labels_after <- c(rep("Source", nrow(Z_s)), rep("Target", nrow(Z_t)))

  pca_after <- switch(
    method,
    pca  = prcomp(data_after, rank. = 2)$x,
    tsne = Rtsne::Rtsne(data_after)$Y
  )
  df_after <- data.frame(PC1 = pca_after[,1],
                         PC2 = pca_after[,2],
                         Label = labels_after)

  p2 <- ggplot2::ggplot(df_after, ggplot2::aes(x = PC1, y = PC2, color = Label)) +
    ggplot2::geom_point(alpha = 0.6) +
    ggplot2::labs(title = paste0("Data Distribution After ", description))

  list(p1 = p1, p2 = p2)
}
