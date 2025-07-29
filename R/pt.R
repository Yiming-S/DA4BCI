
#' Domain Adaptation via Parallel Transport (PT)
#'
#' @description
#' `domain_adaptation_pt` aligns the source domain to the target domain on the
#' cone manifold of symmetric positive–definite (SPD) covariance matrices using
#' the parallel-transport operator proposed by Yair et al. (2019).
#' The linear map
#' \eqn{E = (C_T\,C_S^{-1})^{1/2}} ensures
#' \eqn{E\,C_S\,E^{\mathsf T} = C_T}, thereby preserving Riemannian geometry
#' between domains.
#'
#' @param source_data Numeric matrix (*n* × *p*) of source observations.
#' @param target_data Numeric matrix with the same column dimension (*p*) from
#'   the target domain.
#'
#' @details
#' Procedure:
#' \enumerate{
#'   \item Center both domains by subtracting their column means.
#'   \item Estimate covariances \eqn{C_S} and \eqn{C_T} using the Ledoit–Wolf
#'         shrinkage estimator.
#'   \item Compute the parallel-transport map
#'         \eqn{E = (C_T C_S^{-1})^{1/2}}.
#'   \item Transform centered source features with
#'         \eqn{M = E^{\mathsf T}} so that the resulting covariance equals
#'         \eqn{C_T}.
#'   \item Add back the target mean to obtain the aligned source data.
#' }
#'
#' @return A list with:
#' \describe{
#'   \item{\code{weighted_source_data}}{Source data transported to the target
#'         geometry.}
#'   \item{\code{target_data}}{Unmodified target data (for reference).}
#'   \item{\code{transformation_matrix}}{The mapping \eqn{M = E^{\mathsf T}}.}
#' }
#'
#' @references
#' Yair, O., Ben-Chen, M., & Talmon, R. (2019).
#' *Parallel transport on the cone manifold of SPD matrices for domain
#' adaptation*. \emph{IEEE Transactions on Signal Processing, 67}(7), 1797–1811.
#'
#' @examples
#' \dontrun{
#' set.seed(42)
#' src <- matrix(rnorm(200), nrow = 20)          # 20 × 10
#' tgt <- matrix(rnorm(200) + 1, nrow = 20)      # mean-shifted target
#' pt <- domain_adaptation_pt(src, tgt)
#' }
#'
#' @export
#'

domain_adaptation_pt <- function(source_data, target_data) {

  ## ---------- helper ----------
  center <- function(X, mu) sweep(X, 2L, mu)

  ## 0. Mean alignment ---------------------------------------------------------
  mu_S <- colMeans(source_data)
  mu_T <- colMeans(target_data)
  Xs_c <- center(source_data, mu_S)
  Xt_c <- center(target_data, mu_T)

  ## 1. Ledoit–Wolf shrinkage covariances (SPD) --------------------------------
  C_S <- LW_covariance(Xs_c)
  C_T <- LW_covariance(Xt_c)

  ## 2. Parallel-transport map  E = (C_T C_S^{-1})^{1/2} ------------------------
  E <- matrix_power(C_T %*% solve(C_S), 0.5)      # SPD → SPD, well-defined
  M <- t(E)                                       # right-multiplication map

  ## 3. Transform source & restore target mean ---------------------------------
  Xs_aligned <- Xs_c %*% M
  Xs_aligned <- sweep(Xs_aligned, 2L, mu_T, '+')

  list(
    weighted_source_data  = Xs_aligned,
    target_data           = target_data,
    transformation_matrix = M
  )
}
