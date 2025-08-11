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
#' @param eps  Positive entropy regularization parameter (default \code{0.05}).
#' @param maxit Maximum Sinkhorn iterations (default \code{500}).
#' @param tol  Stopping tolerance on marginal feasibility (default \code{1e-7}).
#' @param cost Cost type: \code{"sqeuclidean"} (default) or \code{"euclidean"}.
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
#' src <- matrix(rnorm(100), nrow = 20, ncol = 5)
#' tgt <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)
#'
#' ot_result <- domain_adaptation_ot(src, tgt, eps = 0.05, maxit = 1000, tol = 1e-8)
#' aligned_source <- ot_result$weighted_source_data
#' aligned_target <- ot_result$target_data
#' dim(aligned_source)  # 20 x 5
#' dim(aligned_target)  # 20 x 5
#' }
#'
#' @references
#' Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport.
#' Advances in Neural Information Processing Systems, 26.
#' Courty, N., Flamary, R., Tuia, D., & Rakotomamonjy, A. (2017).
#' Optimal Transport for Domain Adaptation. IEEE TPAMI, 39(9), 1853–1865.
#'
#' @export
#####################################
domain_adaptation_ot <- function(source_data, target_data,
                                 eps = 0.05, maxit = 500, tol = 1e-7,
                                 cost = c("sqeuclidean", "euclidean")) {

  cost <- match.arg(cost)

  # --- inputs & shape checks ---
  Xs <- as.matrix(source_data); storage.mode(Xs) <- "double"
  Xt <- as.matrix(target_data); storage.mode(Xt) <- "double"
  if (!is.numeric(Xs) || !is.numeric(Xt))
    stop("source_data/target_data must be numeric matrices.")
  if (ncol(Xs) != ncol(Xt))
    stop("source_data/target_data must have the same number of columns.")
  n <- nrow(Xs); m <- nrow(Xt)
  if (n == 0L || m == 0L)
    stop("source_data/target_data must have at least one row.")
  if (!is.numeric(eps) || eps <= 0) stop("'eps' must be positive.")
  maxit <- as.integer(maxit)

  # --- cost matrix C ---
  # Squared Euclidean: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
  XXs <- rowSums(Xs * Xs)
  XXt <- rowSums(Xt * Xt)
  Csq <- outer(XXs, XXt, "+") - 2 * (Xs %*% t(Xt))
  Csq <- pmax(Csq, 0)  # numeric safeguard
  C   <- if (cost == "euclidean") sqrt(Csq) else Csq

  # --- Sinkhorn–Knopp (balanced, uniform marginals) ---
  r <- rep(1 / n, n)   # source marginal
  c <- rep(1 / m, m)   # target marginal

  # Gibbs kernel K = exp(-C/eps) with a tiny floor to avoid zeros
  K <- exp(-C / eps)
  tiny <- .Machine$double.xmin
  K[K < tiny] <- tiny

  # Scaling vectors (always numeric vectors)
  u <- rep(1, n)
  v <- rep(1, m)

  residual <- Inf
  iters <- 0L
  for (iter in seq_len(maxit)) {
    iters <- iter
    Kv <- as.numeric(K %*% v);             Kv[!is.finite(Kv) | Kv <= tiny] <- tiny
    u  <- r / Kv
    Ktu <- as.numeric(t(K) %*% u);          Ktu[!is.finite(Ktu) | Ktu <= tiny] <- tiny
    v  <- c / Ktu

    # Coupling (P = diag(u) K diag(v)) via elementwise outer product
    P <- (u %o% v) * K

    # L-infinity marginal residual
    residual <- max(max(abs(rowSums(P) - r)), max(abs(colSums(P) - c)))
    if (!is.finite(residual) || residual <= tol) break
  }
  converged <- is.finite(residual) && residual <= tol

  # --- barycentric mapping: Xs -> (P Xt) / (P 1) ---
  rs <- rowSums(P); rs[!is.finite(rs) | rs <= tiny] <- tiny
  Xs_map <- sweep(P %*% Xt, 1, rs, "/")

  # --- return ---
  list(
    weighted_source_data = Xs_map,
    target_data          = Xt,
    ot_plan              = P,
    cost                 = C,
    epsilon              = eps,
    iterations           = iters,
    converged            = converged,
    residual             = residual
  )
}
