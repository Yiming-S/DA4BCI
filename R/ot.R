#####################################
#' Entropy-Regularized OT (Sinkhorn–Knopp) with Barycentric Mapping
#'
#' @description
#' The `domain_adaptation_ot` function computes a balanced, entropy-regularized
#' optimal transport (OT) coupling between \code{source_data} and \code{target_data}
#' using the Sinkhorn–Knopp iterations with uniform marginals. It then applies the
#' barycentric mapping \eqn{X_s \mapsto P X_t / (P \mathbf{1})} to align source samples
#' toward the target domain.
#'
#' @param source_data A numeric matrix of size \eqn{n \times d} (source samples by features).
#' @param target_data A numeric matrix of size \eqn{m \times d} (target samples by features).
#' @param control A list of control parameters:
#' \describe{
#'   \item{\code{eps}}{Positive entropy regularization parameter (default \code{0.05}).}
#'   \item{\code{maxit}}{Maximum Sinkhorn iterations (default \code{500}).}
#'   \item{\code{tol}}{Stopping tolerance on marginal feasibility (default \code{1e-7}).}
#'   \item{\code{cost}}{Cost type: \code{"sqeuclidean"} (default) or \code{"euclidean"}.}
#' }
#'
#' @return A list containing:
#' \describe{
#'   \item{\code{weighted_source_data}}{The barycentrically mapped source data \eqn{P X_t / (P \mathbf{1})}.}
#'   \item{\code{target_data}}{The original target data (possibly to be used downstream).}
#'   \item{\code{ot_plan}}{The OT coupling matrix \eqn{P \in \mathbb{R}^{n \times m}}.}
#'   \item{\code{cost}}{The transport cost matrix \eqn{C}.}
#'   \item{\code{epsilon}}{The regularization parameter actually used.}
#'   \item{\code{iterations}}{Number of Sinkhorn iterations performed.}
#'   \item{\code{converged}}{Logical flag indicating whether the marginal residual met \code{tol}.}
#'   \item{\code{residual}}{Final infinity-norm marginal residual.}
#' }
#'
#' @details
#' This function solves the balanced, entropy-regularized OT problem
#' \deqn{\min_{P \ge 0} \langle P, C \rangle + \epsilon \sum_{ij} P_{ij}(\log P_{ij} - 1)}
#' subject to \eqn{P \mathbf{1}_m = r}, \eqn{\mathbf{1}_n^\top P = c^\top}, where
#' \eqn{r = \mathbf{1}_n/n} and \eqn{c = \mathbf{1}_m/m}. The coupling is computed via
#' the Sinkhorn–Knopp updates with kernel \eqn{K = \exp(-C/\epsilon)}. The barycentric
#' map transports each source point to the convex combination of target points
#' weighted by the corresponding row of \eqn{P}.
#'
#' @examples
#' \dontrun{
#' set.seed(123)
#' Xs <- matrix(rnorm(200), nrow = 20, ncol = 5)
#' Xt <- matrix(rnorm(180, mean = 1), nrow = 18, ncol = 5)
#' res <- domain_adaptation_ot(Xs, Xt, eps = 0.05, maxit = 1000, tol = 1e-8)
#' str(res$weighted_source_data)  # 20 x 5 matrix
#' }
#'
#' @references
#' Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport.
#' \emph{Advances in Neural Information Processing Systems}, 26.
#'
#' Courty, N., Flamary, R., Tuia, D., & Rakotomamonjy, A. (2017). Optimal Transport for Domain Adaptation.
#' \emph{IEEE Transactions on Pattern Analysis and Machine Intelligence}, 39(9), 1853–1865.
#'
#'
#' @export
#####################################
domain_adaptation_ot <- function(source_data, target_data,
                                 eps = 0.05, maxit = 500,
                                 tol = 1e-7, cost = "sqeuclidean") {

  Xs <- as.matrix(source_data)
  Xt <- as.matrix(target_data)
  if (!is.numeric(Xs) || !is.numeric(Xt)) {
    stop("source_data and target_data must be numeric matrices.")
  }
  if (ncol(Xs) != ncol(Xt)) {
    stop("source_data and target_data must have the same number of columns (features).")
  }
  n <- nrow(Xs); m <- nrow(Xt)
  if (n == 0L || m == 0L) stop("source_data and target_data must have at least one row.")


  # Step 2: Cost matrix (vectorized)
  # Squared Euclidean: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
  XXs <- rowSums(Xs * Xs)
  XXt <- rowSums(Xt * Xt)
  Csq <- outer(XXs, XXt, "+") - 2 * (Xs %*% t(Xt))
  Csq <- pmax(Csq, 0)  # numerical safety against tiny negatives
  C   <- if (identical(cost, "euclidean")) sqrt(Csq) else Csq

  # Step 3: Sinkhorn–Knopp (balanced, uniform marginals)
  r <- rep(1 / n, n)
  c <- rep(1 / m, m)

  # Gibbs kernel; floor to avoid exact zeros
  K <- exp(-C / eps)
  tiny <- .Machine$double.xmin
  K[K < tiny] <- tiny

  # Initialize scalings
  u <- rep(1, n)
  v <- rep(1, m)

  # Iterate until marginals match within tol
  residual <- Inf
  iters <- 0L
  for (iter in seq_len(maxit)) {
    iters <- iter

    Kv <- K %*% v
    Kv[!is.finite(Kv) | Kv <= tiny] <- tiny
    u <- r / Kv

    Ktu <- t(K) %*% u
    Ktu[!is.finite(Ktu) | Ktu <= tiny] <- tiny
    v <- c / Ktu

    # Current coupling and marginal residual
    P <- (u * K) * rep(v, each = n)
    residual <- max(
      max(abs(rowSums(P) - r)),
      max(abs(colSums(P) - c))
    )
    if (residual <= tol) break
  }
  converged <- residual <= tol

  # Step 4: Barycentric mapping of source toward target
  rs <- rowSums(P) + tiny  # avoid division by zero
  Xs_map <- (P %*% Xt) / rs

  # Step 5: Return results
  return(list(
    weighted_source_data = Xs_map,
    target_data          = Xt,
    ot_plan              = P,
    cost                 = C,
    epsilon              = eps,
    iterations           = iters,
    converged            = converged,
    residual             = residual
  ))
}
