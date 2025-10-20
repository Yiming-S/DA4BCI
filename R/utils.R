
# utils.R
#
#  kmm_weights, label_shift_em, euclidean_alignment
#  proxy_a_distance, distanceSummary, evaluate_shift
#  plot_data_comparison
#  ph_init, ph_update


#' Kernel Mean Matching Weights
#'
#' @description
#' \code{kmm_weights} computes KMM reweighting coefficients for covariate shift
#' by solving a box-constrained QP via \code{quadprog::solve.QP}.
#'
#' @param Xs Numeric matrix (n × d): source samples.
#' @param Xt Numeric matrix (m × d): target samples.
#' @param sigma Optional positive bandwidth; defaults to \code{sigma_med(Xs, Xt)}.
#' @param B Positive upper bound for each weight.
#' @param eps Nonnegative tolerance on total weight; default \code{B / sqrt(n)}.
#'
#' @details
#' With RBF kernels \code{Kss} and \code{Kst} from \code{\link{rbf_kernel}},
#' minimize \eqn{\tfrac{1}{2} w^\top K_{ss} w - \kappa^\top w} subject to
#' \eqn{0 \le w_i \le B} and \eqn{|1^\top w - n| \le \varepsilon}, where
#' \eqn{\kappa=(n/m)K_{st}\mathbf{1}}. A small ridge is added to ensure PD.
#'
#' @return Numeric vector (length n) with weights in \eqn{[0, B]}.
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' Xs <- matrix(rnorm(200), 20, 10); Xt <- matrix(rnorm(150, 1), 15, 10)
#' w <- kmm_weights(Xs, Xt, B = 10); range(w)
#' }
#' @family domain-adaptation
#' @seealso \code{\link{sigma_med}}, \code{\link{distanceSummary}}
#' @export
#'
kmm_weights <- function(Xs, Xt, sigma = NULL, B = 100, eps = NULL) {
  Xs <- as.matrix(Xs); Xt <- as.matrix(Xt)
  n <- nrow(Xs); m <- nrow(Xt)

  # bandwidth: median heuristic if not provided
  if (is.null(sigma)) sigma <- sigma_med(Xs, Xt)
  # tolerance: default per KMM practice
  if (is.null(eps)) eps <- B / sqrt(max(1, n))

  # kernels
  Kss <- rbf_kernel(Xs, Xs, sigma)   # n x n
  Kst <- rbf_kernel(Xs, Xt, sigma)   # n x m
  Kss <- 0.5 * (Kss + t(Kss))        # symmetrize

  # linear term κ = (n/m) * Kst * 1_m
  dvec <- (n / m) * rowSums(Kst)

  # constraints A^T w >= b:
  # 1) w_i >= 0
  # 2) -w_i >= -B            (w_i <= B)
  # 3) 1^T w >= n - eps
  # 4) -1^T w >= -(n + eps)  (1^T w <= n + eps)
  Aeq  <- matrix(1, nrow = 1, ncol = n)
  Amat <- cbind(diag(n), -diag(n), t(Aeq), -t(Aeq))
  bvec <- c(rep(0, n), rep(-B, n), n - eps, -(n + eps))

  # adaptive jitter to ensure PD for quadprog::solve.QP
  jitters <- c(1e-8, 1e-6, 1e-4, 1e-2)
  sol <- NULL
  for (j in jitters) {
    Dmat_try <- 0.5 * (Kss + t(Kss)) + diag(j, n)
    sol <- try(quadprog::solve.QP(Dmat = Dmat_try, dvec = dvec,
                                  Amat = Amat, bvec = bvec, meq = 0),
               silent = TRUE)
    if (!inherits(sol, "try-error")) break
  }
  if (inherits(sol, "try-error"))
    stop("solve.QP failed; adjust 'sigma', 'B', or 'eps'.")

  # clip to [0, B]
  as.numeric(pmax(0, pmin(B, sol$solution)))
}



#' Prior Adjustment under Label Shift via EM
#'
#' @description
#' \code{label_shift_em} estimates target class priors and adjusts posteriors
#' given source-trained posteriors on target data.
#'
#' @param Ps_yx Numeric matrix (n_t × K): source-model posteriors on target samples (rows sum to 1).
#' @param pi_s Numeric length-K: source priors (will be normalized).
#' @param tol Convergence tolerance. Default \code{1e-7}.
#' @param maxit Maximum EM iterations. Default \code{200}.
#'
#' @return List with \code{pi_t} (estimated target priors), \code{P_adj} (adjusted posteriors),
#'   and \code{iter} (iterations used).
#'
#' @examples
#' \dontrun{
#' P <- matrix(runif(50), 10, 5); P <- P/rowSums(P)
#' out <- label_shift_em(P, rep(1/5, 5))
#' str(out)
#' }
#' @family domain-adaptation
#' @export
#'
label_shift_em <- function(Ps_yx, pi_s, tol = 1e-7, maxit = 200) {
  Ps_yx <- as.matrix(Ps_yx)
  K <- ncol(Ps_yx)
  pi_s <- as.numeric(pi_s)
  pi_s <- pi_s / sum(pi_s)
  pi_t <- pi_s

  it <- 0L
  for (it in seq_len(maxit)) {
    # E-step
    R <- sweep(Ps_yx, 2, pi_t / (pi_s + 1e-15), `*`)
    R <- sweep(R, 1, rowSums(R) + 1e-15, `/`)
    # M-step
    pi_new <- colMeans(R)
    if (max(abs(pi_new - pi_t)) < tol) { pi_t <- pi_new; break }
    pi_t <- pi_new
  }

  # Adjust posteriors under estimated priors
  P_adj <- sweep(Ps_yx, 2, pi_t / (pi_s + 1e-15), `*`)
  P_adj <- sweep(P_adj, 1, rowSums(P_adj) + 1e-15, `/`)
  list(pi_t = pi_t, P_adj = P_adj, iter = it)
}



#' Euclidean Alignment (EA) for EEG Trials
#'
#' @description
#' \code{euclidean_alignment} whitens a domain to identity (or recolors to \code{target_R})
#' and applies the same linear transform to each trial.
#'
#' @param trials List of matrices (channels × samples) from one domain.
#' @param target_R Optional SPD (channels × channels). If \code{NULL}, align to identity.
#'
#' @details
#' Each trial \eqn{X} is mapped by \eqn{X' = R_{\mathrm{target}}^{1/2} R_{\mathrm{domain}}^{-1/2} X},
#' where \eqn{R_{\mathrm{domain}}} is the mean covariance over trials.
#'
#' @return List of aligned trials (same shapes as inputs).
#'
#' @examples
#' \dontrun{
#' tr <- replicate(5, matrix(rnorm(32*100), 32, 100), simplify = FALSE)
#' out <- euclidean_alignment(tr)
#' }
#' @family domain-adaptation
#' @export
#'
euclidean_alignment <- function(trials, target_R = NULL) {
  stopifnot(is.list(trials) && length(trials) > 0)

  covs <- lapply(trials, function(X) {
    X <- as.matrix(X)
    S <- X %*% t(X)
    S / max(1, ncol(X) - 1)
  })
  R <- Reduce(`+`, covs) / length(covs)

  mat_pow <- function(A, p) {
    E <- eigen((A + t(A)) / 2, symmetric = TRUE)
    E$vectors %*% diag((pmax(E$values, 1e-12))^p) %*% t(E$vectors)
  }
  Rm12 <- mat_pow(R, -0.5)
  L <- if (is.null(target_R)) diag(nrow(R)) else mat_pow(target_R, 0.5)

  lapply(trials, function(X) L %*% Rm12 %*% as.matrix(X))
}


#' Proxy A-Distance via Ridge LDA with K-Fold CV
#'
#' @description
#' \code{proxy_a_distance} trains a ridge LDA classifier to separate domains and
#' reports PAD \eqn{= 4\,\mathrm{acc} - 2 = 2(1 - 2\,\mathrm{err})}.
#'
#' @param Xs Numeric matrix (n_s × p): source.
#' @param Xt Numeric matrix (n_t × p): target (same number of columns).
#' @param folds Integer K for stratified CV. Default \code{5}.
#' @param ridge Ridge added to pooled covariance. Default \code{1e-3}.
#' @param seed Optional integer seed.
#'
#' @return List with \code{pad} and \code{err} (symmetrized error).
#'
#' @examples
#' \dontrun{
#' Xs <- matrix(rnorm(200), 20, 10); Xt <- matrix(rnorm(200, 1), 20, 10)
#' proxy_a_distance(Xs, Xt)
#' }
#' @family domain-adaptation
#' @export
#'
proxy_a_distance <- function(Xs, Xt, folds = 5, ridge = 1e-3, seed = NULL) {
  Xs <- as.matrix(Xs); Xt <- as.matrix(Xt)
  if (ncol(Xs) != ncol(Xt)) stop("Xs and Xt must have the same number of columns.")
  X <- rbind(Xs, Xt)
  y <- c(rep(0L, nrow(Xs)), rep(1L, nrow(Xt)))  # 0:source, 1:target

  n <- nrow(X)
  if (!is.null(seed)) set.seed(seed)
  idx <- sample.int(n)
  X <- X[idx, , drop = FALSE]; y <- y[idx]

  # --- stratified folds (ensure at least 1 sample per class per fold) ---
  idx0 <- which(y == 0L); idx1 <- which(y == 1L)
  K <- max(2, min(folds, length(idx0), length(idx1)))
  f0 <- split(sample(idx0), rep(1:K, length.out = length(idx0)))
  f1 <- split(sample(idx1), rep(1:K, length.out = length(idx1)))
  folds_idx <- lapply(seq_len(K), function(k) sort(c(f0[[k]], f1[[k]])))

  # --- helpers: standardize & ridge-LDA ---
  std_fit   <- function(A) list(mu = colMeans(A), sd = pmax(apply(A, 2, sd), 1e-8))
  std_apply <- function(A, s) sweep(sweep(A, 2, s$mu, "-"), 2, s$sd, "/")

  lda_fit <- function(A, yy, ridge = 1e-3) {
    A0 <- A[yy == 0L, , drop = FALSE]; A1 <- A[yy == 1L, , drop = FALSE]
    p  <- ncol(A)
    S0 <- if (nrow(A0) > 1) stats::cov(A0) else matrix(0, p, p)
    S1 <- if (nrow(A1) > 1) stats::cov(A1) else matrix(0, p, p)
    Sp <- ((max(nrow(A0)-1,0) * S0 + max(nrow(A1)-1,0) * S1) / max(nrow(A)-2, 1)) + ridge * diag(p)
    iSp <- solve(Sp)
    mu0 <- colMeans(A0); mu1 <- colMeans(A1)
    w <- as.numeric(iSp %*% (mu1 - mu0))
    b <- -0.5 * (crossprod(mu1, iSp %*% mu1) - crossprod(mu0, iSp %*% mu0))
    list(w = w, b = as.numeric(b))
  }
  lda_pred <- function(mod, A) as.integer(drop(A %*% mod$w + mod$b) >= 0)

  # --- K-fold CV prediction ---
  pred <- integer(n); pred[] <- NA_integer_
  for (k in seq_len(K)) {
    te <- folds_idx[[k]]
    tr <- setdiff(seq_len(n), te)
    # If a fold accidentally drops a class in training, skip it
    if (length(unique(y[tr])) < 2L) next
    s   <- std_fit(X[tr, , drop = FALSE])
    Xtr <- std_apply(X[tr, , drop = FALSE], s)
    Xte <- std_apply(X[te, , drop = FALSE], s)
    mdl <- lda_fit(Xtr, y[tr], ridge = ridge)
    pred[te] <- lda_pred(mdl, Xte)
  }

  # remove any NA (from rare skipped folds) and align y/pred
  ok  <- which(!is.na(pred))
  err <- mean(pred[ok] != y[ok])
  err <- min(err, 1 - err)      # symmetric in domain labels
  pad <- 2 * (1 - 2 * err)      # = 4*acc - 2  with acc = 1 - err
  list(pad = pad, err = err)
}


#' Summarise Distribution Distances Between Domains
#'
#' @description
#' \code{distanceSummary} computes selected metrics (PAD, MMD/MMD2, Energy,
#' Wasserstein, Geodesic, Mahalanobis) between \code{source} and \code{target}.
#'
#' @param source Numeric matrix (n_s × p).
#' @param target Numeric matrix (n_t × p).
#' @param sigma Optional bandwidth for MMD; default \code{sigma_med(source, target)}.
#' @param include Character vector of metrics to include.
#' @param format \code{"list"} or \code{"table"}.
#' @param pad_folds,pad_ridge,pad_seed Controls for \code{\link{proxy_a_distance}}.
#'
#' @return Named list or data frame with metric values; when \code{format="table"},
#'   an attribute \code{sigma_used} is attached.
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(100), 20, 5); Y <- matrix(rnorm(100, 1), 20, 5)
#' distanceSummary(X, Y, format = "table")
#' distanceSummary(X, Y, include = c("PAD","MMD2","Energy","MMD",
#'                                   "Wasserstein","Geodesic","Mahalanobis"))
#' }
#' @family domain-adaptation
#' @seealso \code{\link{compute_mmd}}, \code{\link{compute_wasserstein}},
#'   \code{\link{compute_geodesic}}, \code{\link{compute_mahalanobis}}, \code{\link{compute_energy}}
#' @export
distanceSummary <- function(source, target, sigma = NULL,
                            include = c("PAD","MMD2","Energy","MMD",
                                        "Wasserstein","Geodesic","Mahalanobis"),
                            format = c("list","table"),
                            pad_folds = 5, pad_ridge = 1e-3, pad_seed = NULL) {

  format <- match.arg(format)

  # Helper: coerce to numeric matrix
  .as_mat_num <- function(x, nm) {
    if (is.data.frame(x)) x <- data.matrix(x)
    x <- as.matrix(x)
    storage.mode(x) <- "double"
    if (!is.numeric(x)) stop(sprintf("%s must be numeric.", nm), call. = FALSE)
    x
  }

  source <- .as_mat_num(source, "source")
  target <- .as_mat_num(target, "target")

  if (ncol(source) != ncol(target))
    stop(sprintf("source/target must have same #cols (got %d vs %d).",
                 ncol(source), ncol(target)), call. = FALSE)

  # Kernel bandwidth for MMD
  if (is.null(sigma)) sigma <- DA4BCI::sigma_med(source, target)

  # Proxy A-distance (PAD)
  pad  <- proxy_a_distance(source, target, folds = pad_folds,
                           ridge = pad_ridge, seed = pad_seed)$pad

  # MMD^2 and MMD (RBF)
  mmd2 <- DA4BCI::compute_mmd(source, target, sigma)
  mmd  <- sqrt(max(mmd2, 0))

  # Energy distance (prefer existing implementation)
  energy <- if (exists("compute_energy", mode = "function")) {
    get("compute_energy")(source, target)
  } else {
    # Minimal fallback using Euclidean pairwise distances
    pairwise_euclid <- function(A, B) {
      aa <- rowSums(A^2)
      bb <- rowSums(B^2)
      D2 <- outer(aa, bb, "+") - 2 * (A %*% t(B)); D2[D2 < 0] <- 0; sqrt(D2)
    }
    mean_offdiag <- function(D) {
      n <- nrow(D); if (n <= 1) return(0)
      sum(D[row(D) != col(D)]) / (n * (n - 1))
    }
    Dxy <- pairwise_euclid(source, target)
    Dxx <- pairwise_euclid(source, source)
    Dyy <- pairwise_euclid(target, target)
    2 * mean(Dxy) - mean_offdiag(Dxx) - mean_offdiag(Dyy)
  }

  # Wasserstein-1
  wfun <- if (exists("compute_wasserstein", mode = "function")) get("compute_wasserstein") else NULL
  wass <- if (!is.null(wfun)) wfun(source, target) else NA_real_

  # Geodesic subspace distance
  gfun <- if (exists("compute_geodesic", mode = "function")) get("compute_geodesic") else NULL
  geod <- if (!is.null(gfun)) gfun(source, target) else NA_real_

  # Mahalanobis mean-distance
  mlfun <- if (exists("compute_mahalanobis", mode = "function")) get("compute_mahalanobis") else NULL
  maha  <- if (!is.null(mlfun)) {
    # Defaults: pooled covariance, no shrinkage, non-squared distance
    mlfun(source, target)
  } else {
    # Robust fallback without external dependency
    .mahal_fallback <- function(X, Y, ridge = 1e-6) {
      mux <- colMeans(X); muy <- colMeans(Y)
      Sx  <- stats::cov(X);  Sy  <- stats::cov(Y)
      nx  <- nrow(X); ny <- nrow(Y); p <- ncol(X)
      S <- if (nx + ny - 2 > 0) ((nx - 1) * Sx + (ny - 1) * Sy) / (nx + ny - 2) else Sx
      S <- 0.5 * (S + t(S))
      trp <- sum(diag(S)) / p
      S <- S + ridge * trp * diag(p)
      eig <- eigen(S, symmetric = TRUE)
      eps <- .Machine$double.eps^0.5
      eig$values[eig$values < eps] <- eps
      SInv <- eig$vectors %*% (diag(1 / eig$values)) %*% t(eig$vectors)
      d <- mux - muy
      sqrt(as.numeric(t(d) %*% SInv %*% d))
    }
    .mahal_fallback(source, target)
  }

  # Collect and filter
  all_vals <- c(PAD = pad,
                MMD2 = mmd2,
                Energy = energy,
                MMD = mmd,
                Wasserstein = wass,
                Geodesic = geod,
                Mahalanobis = maha)

  # Keep only requested metrics, preserving order in `include`
  keep <- intersect(include, names(all_vals))
  out_vals <- all_vals[keep]

  if (format == "list") {
    as.list(out_vals)
  } else {
    out <- data.frame(Metric = names(out_vals), Value = as.numeric(out_vals), row.names = NULL)
    attr(out, "sigma_used") <- sigma
    out
  }
}



#' Evaluate Distribution Shift Before/After Adaptation
#'
#' @description
#' \code{evaluate_shift} reports MMD and Wasserstein distances for original data
#' and adapted data.
#'
#' @param source_data Numeric matrix (n_s × p) before adaptation.
#' @param target_data Numeric matrix (n_t × p) before adaptation.
#' @param adapted_source Numeric matrix: adapted source.
#' @param adapted_target Numeric matrix: adapted target.
#'
#' @return Data frame with columns \code{Metric}, \code{Before}, and \code{After}.
#'
#' @examples
#' \dontrun{
#' A <- matrix(rnorm(200), nrow = 20, ncol = 10)
#' B <- matrix(rnorm(200, mean = 2), nrow = 20, ncol = 10)
#' As <- matrix(rnorm(200), nrow = 20, ncol = 10)
#' Bs <- matrix(rnorm(200, mean = 1.5), nrow = 20, ncol = 10)
#' evaluate_shift(A, B, As, Bs)
#' }
#' @family domain-adaptation
#' @seealso \code{\link{compute_mmd}}, \code{\link{compute_wasserstein}}
#' @export
#'
evaluate_shift <- function(source, target,
                           adapted_source, adapted_target) {
  mmd_before <- compute_mmd(source, target, sigma = 1)
  mmd_after <- compute_mmd(adapted_source, adapted_target, sigma = 1)

  wasserstein_before <- compute_wasserstein(source, target)
  wasserstein_after <- compute_wasserstein(adapted_source, adapted_target)

  data.frame(
    Metric = c("MMD", "Wasserstein"),
    Before = c(mmd_before, wasserstein_before),
    After = c(mmd_after, wasserstein_after)
  )
}



#' Visual Comparison Before/After Domain Adaptation
#'
#' @description
#' \code{plot_data_comparison} reduces to 2D via PCA or t-SNE and plots source/target
#' distributions before and (optionally) after adaptation.
#'
#' @param source_data Numeric matrix (n_s × p) before adaptation.
#' @param target_data Numeric matrix (n_t × p) before adaptation.
#' @param Z_s Optional numeric matrix: adapted source.
#' @param Z_t Optional numeric matrix: adapted target.
#' @param description Character string appended to plot titles.
#' @param method Either \code{"pca"} (default) or \code{"tsne"}.
#'
#' @details
#' Concatenates domains, applies \code{\link[stats]{prcomp}} or \code{Rtsne::Rtsne},
#' and returns \code{ggplot2} scatter plots.
#'
#' @return List with \code{p1} (before) and optionally \code{p2} (after), both \code{ggplot} objects.
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(100), 20, 5); Y <- matrix(rnorm(100, 2), 20, 5)
#' out <- plot_data_comparison(X, Y, method = "pca"); print(out$p1)
#' }
#' @family viz-monitoring
#' @export
#'
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


#' Page–Hinkley Change Detector: Initialize State
#'
#' @description
#' \code{ph_init} initializes the internal state for the Page–Hinkley change
#' detection test with exponential moving average.
#'
#' @param delta Insensitivity parameter for mean deviation.
#' @param lambda Detection threshold.
#' @param alpha Exponential moving average factor for the mean.
#'
#' @return List state to be passed to \code{\link{ph_update}}.
#'
#' @examples
#' \dontrun{
#' s <- ph_init()
#' }
#' @family viz-monitoring
#' @export
#'
ph_init <- function(delta = 0.005, lambda = 50, alpha = 0.999) {
  list(mean = 0, cum = 0, min_cum = 0,
       delta = delta, lambda = lambda,
       alpha = alpha)
}



#' Page–Hinkley Change Detector: Update Step
#'
#' @description
#' \code{ph_update} updates the Page–Hinkley state with a new observation and
#' indicates whether a change is detected.
#'
#' @param state State list returned by \code{\link{ph_init}}.
#' @param x New scalar observation.
#'
#' @return List with \code{state} (updated) and \code{change} (logical flag).
#'
#' @examples
#' \dontrun{
#' s <- ph_init()
#' for (z in rnorm(100)) { tmp <- ph_update(s, z); s <- tmp$state }
#' }
#' @family viz-monitoring
#' @export
#'
ph_update <- function(state, x) {
  state$mean <- state$alpha * state$mean + (1 - state$alpha) * x
  state$cum  <- state$cum + (x - state$mean - state$delta)
  state$min_cum <- min(state$min_cum, state$cum)
  changed <- (state$cum - state$min_cum) > state$lambda
  list(state = state, change = changed)
}




# helper: define %||% if not already defined
`%||%` <- function(a, b) if (!is.null(a)) a else b








