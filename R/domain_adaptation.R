#' Unified Interface for Domain Adaptation Methods
#'
#' @description
#' Provides a single entry point to apply one of several domain adaptation techniques,
#' particularly in EEG-based scenarios. Users can choose among TCA (Transfer Component
#' Analysis), SA (Subspace Alignment), MIDA (Maximum Independence Domain Adaptation), RD
#' (Riemannian Distance), CORAL (Correlation Alignment), or GFK (Geodesic Flow Kernel).
#' The method-specific parameters are passed via the \code{control} list.
#'
#' @param source_data A numeric matrix (or data frame) representing the source domain, with
#' rows as observations and columns as features.
#' @param target_data A numeric matrix (or data frame) representing the target domain, with
#' the same number of columns as \code{source_data}.
#' @param method A character string specifying the chosen domain adaptation method. Valid
#' choices are \code{"tca"}, \code{"sa"}, \code{"mida"}, \code{"rd"}, \code{"coral"}, or \code{"gfk"}.
#' Defaults to \code{"sa"}.
#' @param control A named list of additional parameters passed to the selected method. For example:
#' \itemize{
#'   \item \code{tca}: \code{k}, \code{sigma}, \code{mu}.
#'   \item \code{sa}: \code{k}.
#'   \item \code{mida}: \code{k}, \code{sigma}, \code{mu}, \code{max}.
#'   \item \code{rd}: typically none (covariances computed internally).
#'   \item \code{coral}: \code{lambda}.
#'   \item \code{gfk}: \code{dim_subspace}.
#' }
#'
#' @details
#' This function checks the argument \code{method} and dispatches the input data to one of several
#' domain adaptation routines:
#'
#' \describe{
#'   \item{\strong{TCA}}{Transfer Component Analysis. Projects source and target data onto a latent
#'   space to reduce distribution mismatch.}
#'   \item{\strong{SA}}{Subspace Alignment. Learns a linear alignment between source and target
#'   principal components.}
#'   \item{\strong{MIDA}}{Maximum/Minimum Independence Domain Adaptation. Adjusts feature
#'   representations to control (maximize or minimize) domain dependence.}
#'   \item{\strong{RD}}{Riemannian Distance. Uses rotation based on covariance matrices to align
#'   source and target.}
#'   \item{\strong{CORAL}}{Correlation Alignment. Matches second-order statistics (covariance) via
#'   whitening and re-coloring.}
#'   \item{\strong{GFK}}{Geodesic Flow Kernel. Interpolates between source and target subspaces
#'   along the Grassmann manifold.}
#' }
#'
#' @return A list whose structure depends on the selected \code{method}. Typically, it includes:
#' \describe{
#'   \item{\code{weighted_source_data}}{The adapted source data in the learned or aligned space.}
#'   \item{\code{target_data}}{A transformed version of the target data (if relevant to the method).}
#'   \item{\code{rotation_matrix}}{(RD only) The rotation used to align source and target covariances.}
#' }
#'
#' @references
#' Lan, Z., Sourina, O., Wang, L., Scherer, R., \& Müller-Putz, G. R. (2018).
#' \emph{Domain adaptation techniques for EEG-based emotion recognition: a comparative study
#' on two public datasets.} IEEE Transactions on Cognitive and Developmental Systems, 11(1), 85--94.
#'
#' Sun, B. \& Saenko, K. (2016). \emph{Correlation Alignment for Unsupervised Domain Adaptation.}
#' In: \emph{Domain Adaptation in Computer Vision Applications}, 153--171, Springer.
#'
#' Gong, B., Shi, Y., Sha, F., \& Grauman, K. (2012).
#' \emph{Geodesic Flow Kernel for Unsupervised Domain Adaptation.} In: \emph{CVPR}, 2066--2073, IEEE.
#'
#' @seealso \code{\link{domain_adaptation_tca}}, \code{\link{domain_adaptation_sa}},
#'   \code{\link{domain_adaptation_mida}}, \code{\link{domain_adaptation_riemannian}},
#'   \code{\link{domain_adaptation_coral}}, \code{\link{domain_adaptation_gfk}}
#'
#' @examples
#' \dontrun{
#' # Simulate data
#' set.seed(123)
#' source_mat <- matrix(rnorm(100), nrow = 20, ncol = 5)
#' target_mat <- matrix(rnorm(100, mean = 2), nrow = 20, ncol = 5)
#'
#' # Example 1: Apply SA (default)
#' res_sa <- domain_adaptation(source_mat, target_mat, method = "sa",
#'                             control = list(k = 3))
#'
#' # Example 2: Apply TCA
#' res_tca <- domain_adaptation(source_mat, target_mat, method = "tca",
#'                              control = list(k = 5, sigma = 1, mu = 0.1))
#'
#' # Check results
#' dim(res_sa$weighted_source_data)
#' dim(res_tca$weighted_source_data)
#' }
#'
#' @export


domain_adaptation <- function(source_data, target_data,
                              method = "sa",
                              control = list()) {
  method <- match.arg(method, choices = c("tca", "sa", "mida", "rd", "coral", "gfk"))

  if (method == "tca") {
    k <- ifelse(!is.null(control$k), control$k, 10)
    sigma <- ifelse(!is.null(control$sigma), control$sigma, 1)
    mu <- ifelse(!is.null(control$mu), control$mu, 1)
    return(domain_adaptation_tca(source_data, target_data, k = k, sigma = sigma, mu = mu))

  } else if (method == "sa") {
    k <- ifelse(!is.null(control$k), control$k, 10)
    return(domain_adaptation_sa(source_data, target_data, k = k))

  } else if (method == "mida") {
    k     <- ifelse(!is.null(control$k),     control$k,     10)
    sigma <- ifelse(!is.null(control$sigma), control$sigma, 1)
    mu    <- ifelse(!is.null(control$mu),    control$mu,    0.1)
    max   <- ifelse(!is.null(control$max),   control$max,   TRUE)
    return(domain_adaptation_mida(source_data, target_data,
                                  k = k, sigma = sigma, mu = mu, max = max))

  } else if (method == "rd") {
    cov_source <- cov(source_data)
    cov_target <- cov(target_data)
    alignment_result <- domain_adaptation_riemannian(cov_source, cov_target)
    cov_source_aligned <- alignment_result$C_source_aligned
    rotation_matrix    <- alignment_result$rotation_matrix
    return(list(weighted_source_data = source_data %*% rotation_matrix,
                target_data         = target_data,
                cov_source_aligned  = cov_source_aligned,
                rotation_matrix     = rotation_matrix))

  } else if (method == "coral") {
    lambda <- ifelse(!is.null(control$lambda), control$lambda, 1e-5)
    return(domain_adaptation_coral(source_data, target_data, lambda))

  } else if (method == "gfk") {
    dim_subspace <- ifelse(!is.null(control$dim_subspace), control$dim_subspace, 10)
    return(domain_adaptation_gfk(source_data, target_data, dim_subspace))
  }
}

