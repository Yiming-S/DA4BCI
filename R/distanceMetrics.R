
# compute_distance_matrix
# compute_wasserstein
# compute_mmd
# compute_energy

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

  d2 <- outer(source_norms, target_norms, "+") - 2 * cross_term  # squared distances

  # Precision safeguard
  d2[d2 > -eps & d2 < 0] <- 0                   # clamp tiny negatives to zero
  d2[d2 < -eps]         <- NA_real_            # flag unexpected large negatives

  sqrt(d2)
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

####################################
#' Compute Energy Distance
#'
#' @description
#' The \code{compute_energy} function calculates the (squared-rooted) Energy
#' Distance between the empirical distributions represented by
#' \code{source} and \code{target}.  Energy Distance is a metric that, for
#' continuous data, is equivalent to Maximum Mean Discrepancy (MMD) with
#' a distance‐induced kernel and is widely used for two-sample testing and
#' domain-adaptation diagnostics.
#'
#' @param source A numeric matrix whose rows are observations and columns are
#'   features (source domain).
#' @param target A numeric matrix with the same column structure as
#'   \code{source} (target domain).
#'
#' @return A single numeric value giving the Energy Distance between
#'   \code{source} and \code{target}.
#'
#' @details
#' Let \eqn{X,\,X'} be i.i.d.\ samples from the source distribution and
#' \eqn{Y,\,Y'} from the target distribution.  The population Energy
#' Distance is defined as
#' \deqn{D_E = \sqrt{2\,\mathbb{E}\|X-Y\|
#'   - \mathbb{E}\|X-X'\|
#'   - \mathbb{E}\|Y-Y'\|}\,.}
#' The estimator implemented here replaces the expectations by the
#' corresponding empirical means of pairwise Euclidean distances.  Numerical
#' precision is enforced by clamping negative arguments of the final square
#' root to zero.
#'
#' @examples
#' \dontrun{
#' set.seed(42)
#' src <- matrix(rnorm(100),  nrow = 20, ncol = 5)
#' tgt <- matrix(rnorm(100, 1), nrow = 20, ncol = 5)
#' ed <- compute_energy(src, tgt)
#' cat("Energy Distance:", ed, "\n")
#' }
#'
#' @export
####################################
compute_energy <- function(source, target) {
  # Intra-domain pairwise distances
  ds   <- compute_distance_matrix(source, source)
  dt   <- compute_distance_matrix(target, target)

  # Inter-domain pairwise distances
  d_st <- compute_distance_matrix(source, target)

  # Empirical estimator (Székely et al.)
  ed2 <- 2 * mean(d_st) - mean(ds) - mean(dt)

  # Guard against tiny negative values due to numerical error
  sqrt(pmax(ed2, 0))
}


####################################
#' Summarise Wasserstein, MMD, and Energy distances
#'
#' @description
#' \code{distanceSummary} computes three widely-used continuous–distribution
#' metrics—Wasserstein distance, Maximum Mean Discrepancy (MMD), and Energy
#' Distance—between \code{source} and \code{target}, and returns them in a tidy
#' data frame.
#'
#' @param source A numeric matrix (observations × features) representing the
#'   source domain.
#' @param target A numeric matrix with the same column structure as
#'   \code{source} (target domain).
#' @param sigma  Optional positive scalar bandwidth for MMD.  When
#'   \code{NULL} (default), the median heuristic \code{sigma_med(source,target)}
#'   is used.
#'
#' @return A data frame with two columns:
#'   \itemize{
#'     \item \code{Metric} – character string: "Wasserstein", "MMD", "Energy"
#'     \item \code{Value}  – numeric distance value (smaller is better)
#'   }
#'
#' @details
#' \strong{Interpretation}\cr
#' All three metrics are non-negative and equal to zero when the two empirical
#' distributions coincide.  In domain-adaptation pipelines a successful
#' adaptation step should drive these values downward while improving
#' classification accuracy on the target domain.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' src <- matrix(rnorm(100), 20, 5)
#' tgt <- matrix(rnorm(100, 1), 20, 5)
#'
#' distanceSummary(src, tgt)
#' }
#'
#' @export
####################################
distanceSummary <- function(source, target, sigma = NULL) {
  # Select bandwidth for MMD if not provided
  if (is.null(sigma)) sigma <- sigma_med(source, target)

  w <- compute_wasserstein(source, target)
  m <- compute_mmd(source, target, sigma)
  e <- compute_energy(source, target)

  data.frame(
    Metric = c("Wasserstein", "MMD", "Energy"),
    Value  = c(w, m, e),
    row.names = NULL
  )
}
