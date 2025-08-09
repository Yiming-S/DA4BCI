
# ---- Entropy ----
entropyTarget <- function(ps) {
  ps <- pmax(ps, 1e-12)
  ps <- ps / rowSums(ps)
  -mean(rowSums(ps * log(ps)))
}

entropyTargetAdjusted <- function(ptPost, ySource) {
  K <- ncol(ptPost)
  piS <- as.numeric(table(factor(ySource, levels = seq_len(K))))
  piS <- piS / sum(piS)
  adj <- label_shift_em(Ps_yx = ptPost, pi_s = piS)  # external
  entropyTarget(adj$P_adj)
}

# ---- Reverse validation (binary) ----
reverseValidation <- function(Xs, y, Xt, folds = 5,
                              clfFun  = function(df) stats::glm(y ~ ., data = df, family = stats::binomial()),
                              predFun = function(fit, X) stats::predict(fit, newdata = data.frame(X), type = "response")) {
  y <- as.integer(y); if (!all(y %in% c(0L, 1L))) y <- as.integer(factor(y)) - 1L
  n <- nrow(Xs); f <- sample(rep(seq_len(folds), length.out = n))
  errs <- numeric(folds)
  for (k in seq_len(folds)) {
    tr <- f != k; va <- !tr
    fitST <- clfFun(data.frame(y = y[tr], Xs[tr, , drop = FALSE]))
    pT    <- predFun(fitST, Xt)
    yhatT <- as.integer(pT >= 0.5)
    fitTS <- clfFun(data.frame(y = yhatT, Xt))
    pSV   <- predFun(fitTS, Xs[va, , drop = FALSE])
    yhat  <- as.integer(pSV >= 0.5)
    errs[k] <- mean(yhat != y[va])
  }
  mean(errs)
}

# ---- KMM diagnostics ----
kmmDiag <- function(Xs, Xt) {
  # If kmm_weights() is missing, return NAs
  if (!exists("kmm_weights", mode = "function"))
    return(setNames(rep(NA_real_, 5L), c("kmmCv","kmmMax","kmmQ95","kmmQ99","kmmEssInv")))
  obj <- kmm_weights(Xs, Xt)  # external
  w <- NULL
  if (is.numeric(obj)) {
    w <- as.numeric(obj)
  } else if (is.list(obj)) {
    for (nm in c("w","weights","alpha","beta")) if (!is.null(obj[[nm]])) { w <- as.numeric(obj[[nm]]); break }
  } else if (!is.null(attr(obj, "weights"))) {
    w <- as.numeric(attr(obj, "weights"))
  }
  if (is.null(w)) return(setNames(rep(NA_real_, 5L), c("kmmCv","kmmMax","kmmQ95","kmmQ99","kmmEssInv")))
  w <- w[is.finite(w)]
  if (!length(w)) return(setNames(rep(NA_real_, 5L), c("kmmCv","kmmMax","kmmQ95","kmmQ99","kmmEssInv")))
  m <- mean(w); s <- stats::sd(w)
  q <- stats::quantile(w, probs = c(0.95, 0.99), names = FALSE, type = 7)
  ess <- (sum(w))^2 / sum(w^2)
  setNames(
    c(if (m > 0) s/m else NA_real_, max(w), q[1], q[2], if (is.finite(ess) && ess > 0) 1/ess else NA_real_),
    c("kmmCv","kmmMax","kmmQ95","kmmQ99","kmmEssInv")
  )
}

# ---- DA transforms ----
daGetTransforms <- function(method, Xs, Xt, control = list(), ys = NULL) {
  method  <- tolower(method)
  control <- normalizeControl(method, control)
  if (method == "m3d" && is.null(control$source_labels)) {
    if (is.null(ys)) stop("method='m3d' requires control$source_labels or argument 'ys'.")
    control$source_labels <- ys
  }
  res <- domain_adaptation(  # external
    source_data = Xs,
    target_data = Xt,
    method      = method,
    control     = control
  )
  list(XsTr = res$weighted_source_data, XtTr = res$target_data, raw = res)
}

# ---- Evaluate one candidate (binary) ----
daEval <- function(method, Xs, y, Xt, control = list(),
                   entropyAdjust = FALSE, foldsRV = 5,
                   clfFun  = function(df) stats::glm(y ~ ., data = df, family = stats::binomial()),
                   predFun = function(fit, X) stats::predict(fit, newdata = data.frame(X), type = "response")) {
  tr    <- daGetTransforms(method, Xs, Xt, control, ys = y)
  sh    <- shift_report(tr$XsTr, tr$XtTr)  # external
  fit   <- clfFun(data.frame(y = as.integer(y), tr$XsTr))
  pT    <- predFun(fit, tr$XtTr)
  ptMat <- cbind(1 - pT, pT)
  ent   <- if (entropyAdjust) entropyTargetAdjusted(ptMat, y) else entropyTarget(ptMat)
  rv    <- reverseValidation(tr$XsTr, y, tr$XtTr, folds = foldsRV, clfFun = clfFun, predFun = predFun)
  kd    <- kmmDiag(tr$XsTr, tr$XtTr)

  ctrlDf <- if (length(control)) as.data.frame(control, optional = TRUE) else data.frame()
  cbind(
    data.frame(method = method, stringsAsFactors = FALSE),
    ctrlDf,
    data.frame(PAD = sh$PAD, MMD2 = sh$MMD2, Energy = sh$Energy, Entropy = ent, rv = rv, stringsAsFactors = FALSE),
    as.data.frame(as.list(kd), optional = TRUE)
  )
}

# ---- Rank aggregation (smaller is better) ----
rankAggregate <- function(tab, weights) {
  w <- weights[names(weights) %in% names(tab)]
  R <- apply(tab[, names(w), drop = FALSE], 2, rank, ties.method = "average")
  as.numeric(R %*% matrix(w, ncol = 1))
}

# ---- Model selection over candidates ----
# candidates: list(list(method = "...", control = list(...)), ...)
daSelect <- function(candidates, Xs, y, Xt,
                     entropyAdjust = FALSE, foldsRV = 5,
                     weights = c(PAD = 1, MMD2 = 1, Energy = 1, Entropy = 1, rv = 2)) {
  rows <- lapply(seq_along(candidates), function(i) {
    ci <- candidates[[i]]
    df <- daEval(ci$method, Xs, y, Xt, control = ci$control, entropyAdjust = entropyAdjust, foldsRV = foldsRV)
    df$id <- i
    df
  })
  res <- do.call(rbind, rows)
  res$score <- rankAggregate(res, weights)
  res[order(res$score), , drop = FALSE]
}

# ---- Control normalization (unified aliases) ----
normalizeControl <- function(method, control) {
  method <- tolower(method); ctl <- control
  alias <- function(dst, ...) {
    al <- list(...)
    for (nm in al) if (is.null(ctl[[dst]]) && !is.null(ctl[[nm]])) ctl[[dst]] <- ctl[[nm]]
  }
  if (method == "tca") {
    alias("k","q","n_components","dim","dim_subspace")
    alias("sigma","bandwidth")
    alias("mu","lambda","reg","alpha","ridge")
  } else if (method == "sa") {
    alias("k","q","n_components","dim")
  } else if (method == "mida") {
    alias("k","q","n_components","dim"); alias("sigma","bandwidth"); alias("mu","lambda","reg","alpha","ridge")
    if (is.null(ctl$max)) ctl$max <- TRUE
  } else if (method == "gfk") {
    alias("dim_subspace","q","n_components","dim")
  } else if (method == "coral") {
    alias("lambda","reg","alpha","ridge")
  } else if (method == "m3d") {
    normStage <- function(st) {
      if (is.null(st$method)) return(st)
      sm <- tolower(st$method); sc <- st$control; if (is.null(sc)) sc <- list()
      st$control <- normalizeControl(sm, sc); st
    }
    if (!is.null(ctl$stage1)) ctl$stage1 <- normStage(ctl$stage1)
    if (!is.null(ctl$stage2)) ctl$stage2 <- normStage(ctl$stage2)
  }
  ctl
}
