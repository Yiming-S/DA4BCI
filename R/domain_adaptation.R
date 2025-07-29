
#' Unified Interface for Domain Adaptation Methods
#' @name domain_adaptation
#' @aliases domain_adaptation-method
#'
#' @description
#' Provides a single entry point to apply one of several domain adaptation techniques,
#' particularly in EEG-based scenarios. Users can choose among TCA (Transfer Component
#' Analysis), SA (Subspace Alignment), MIDA (Maximum Independence Domain Adaptation),
#' RD (Riemannian Distance), CORAL (Correlation Alignment), GFK (Geodesic Flow Kernel),
#' ART (Aligned Riemannian Transport), PT (Parallel Transport), or M3D (Manifold-based
#' multi-step Domain adaptation with Dynamic distribution).
#' The method-specific parameters are passed via the \code{control} list.
#'
#' @param source_data A numeric matrix (or data frame) representing the source domain, with
#'   rows as observations and columns as features.
#' @param target_data A numeric matrix (or data frame) representing the target domain, with
#'   the same number of columns as \code{source_data}.
#' @param method A character string specifying the chosen domain adaptation method. Valid
#'   choices are \code{"tca"}, \code{"sa"}, \code{"mida"}, \code{"rd"}, \code{"coral"},
#'   \code{"gfk"}, \code{"art"}, \code{"pt"}, or \code{"m3d"}. Defaults to \code{"sa"}.
#' @param control A named list of additional parameters passed to the selected method. For example:
#' \itemize{
#'   \item \code{tca}: \code{k}, \code{sigma}, \code{mu}
#'   \item \code{sa}: \code{k}
#'   \item \code{mida}: \code{k}, \code{sigma}, \code{mu}, \code{max}
#'   \item \code{rd}: (none; covariances computed internally)
#'   \item \code{coral}: \code{lambda}
#'   \item \code{gfk}: \code{dim_subspace}
#'   \item \code{art}: (none; two-argument ART routine)
#'   \item \code{pt}:  (none; two-argument PT routine)
#'   \item \code{m3d}: \code{source_labels}, \code{stage1}, \code{stage2},
#'                     \code{l_iter}, \code{lambda_ridge}, \code{eta_kernel},
#'                     \code{label_offset}, \code{expl_var}, \code{max_dim}
#' }
#'
#' @details
#' This function checks the argument \code{method} and dispatches the input data
#' to one of several domain adaptation routines:
#'
#' \describe{
#'   \item{\strong{TCA}}{Transfer Component Analysis: projects source and target
#'   data onto a latent space to reduce distribution mismatch.}
#'   \item{\strong{SA}}{Subspace Alignment: learns a linear alignment between
#'   source and target principal components.}
#'   \item{\strong{MIDA}}{Maximum (or Minimum) Independence Domain Adaptation:
#'   adjusts feature representations to control domain dependence.}
#'   \item{\strong{RD}}{Riemannian Distance alignment using covariance rotation.}
#'   \item{\strong{CORAL}}{Correlation Alignment: matches second-order statistics
#'   (covariance) via whitening and re-coloring.}
#'   \item{\strong{GFK}}{Geodesic Flow Kernel: interpolates between source and
#'   target subspaces along the Grassmann manifold.}
#'   \item{\strong{ART}}{Aligned Riemannian Transport: aligns covariances by
#'   whitening at the source and coloring with the target SPD geometry.}
#'   \item{\strong{PT}}{Parallel Transport: exact Riemannian alignment via the
#'   map \eqn{E=(C_T C_S^{-1})^{1/2}}}
#'   \item{\strong{M3D}}{Manifold-based multi-step Domain adaptation with Dynamic
#'   distribution: two-stage alignment followed by iterative kernel refinement.}
#' }
#'
#' @return A list whose structure depends on the selected \code{method}. Typically, it includes:
#' \describe{
#'   \item{\code{weighted_source_data}}{The adapted source data in the learned or aligned space.}
#'   \item{\code{target_data}}{A transformed version of the target data (if relevant).}
#'   \item{\code{rotation_matrix}}{(RD only) The rotation used to align covariances.}
#'   \item{\code{transformation_matrix}}{(ART/PT) The linear map applied to the source data.}
#' }
#'
#' @references
#' Yair, O., Ben-Chen, M., & Talmon, R. (2019).
#' *Parallel transport on the cone manifold of SPD matrices for domain adaptation.*
#' IEEE Transactions on Signal Processing, 67(7), 1797–1811.
#'
#' Luo, T., Zhang, J., Qiu, Y., *et al.* (2024).
#' *M3D: Manifold-based domain adaptation with dynamic distribution for non-deep
#' transfer learning in cross-subject and cross-session EEG-based emotion
#' recognition.* arXiv:2404.15615.
#'
#' Lan, Z., Sourina, O., Wang, L., Scherer, R., & Müller-Putz, G. R. (2018).
#' *Domain adaptation techniques for EEG-based emotion recognition: a comparative
#' study on two public datasets.* IEEE Transactions on Cognitive and Developmental
#' Systems, 11(1), 85–94.
#'
#' Sun, B., & Saenko, K. (2016). *Correlation Alignment for Unsupervised Domain
#' Adaptation.* In *Domain Adaptation in Computer Vision Applications* (pp. 153–171).
#'
#' Gong, B., Shi, Y., Sha, F., & Grauman, K. (2012). *Geodesic Flow Kernel for
#' Unsupervised Domain Adaptation.* In *CVPR* (pp. 2066–2073). IEEE.
#'
#' Farahani, A., Voghoei, S., Rasheed, K., & Arabnia, H. R. (2021).
#' *A brief review of domain adaptation.* In *Advances in Data Science and
#' Information Engineering* (pp. 877–894).
#'
#' Zanini, P., Congedo, M., Jutten, C., Said, S., & Berthoumieu, Y. (2017)
#' *Transfer learning: A Riemannian geometry framework with applications to brain–computer interfaces.*.
#' IEEE Transactions on Biomedical Engineering, 65(5), 1107-1116.
#'
#'
#' @seealso \code{\link{domain_adaptation_tca}}, \code{\link{domain_adaptation_sa}},
#'   \code{\link{domain_adaptation_mida}}, \code{\link{domain_adaptation_riemannian}},
#'   \code{\link{domain_adaptation_coral}}, \code{\link{domain_adaptation_gfk}},
#'   \code{\link{domain_adaptation_art}}, \code{\link{domain_adaptation_pt}},
#'   \code{\link{domain_adaptation_m3d}}
#'
#' @examples
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
#'
#' @export


# helper: a %||% b   → b is used only when a is NULL
`%||%` <- function(a, b) if (!is.null(a)) a else b


domain_adaptation <- function(source_data,
                              target_data,
                              method  = "sa",
                              control = list()) {

  method <- match.arg(method,
                      choices = c("tca", "sa", "mida", "rd",
                                  "coral", "gfk", "art", "pt", "m3d"))

  #TCA
  if (method == "tca") {
    k     <- control$k     %||% 10
    sigma <- control$sigma %||% 1
    mu    <- control$mu    %||% 1
    return(domain_adaptation_tca(source_data, target_data,
                                 k = k, sigma = sigma, mu = mu))

    # SA
  } else if (method == "sa") {
    k <- control$k %||% 10
    return(domain_adaptation_sa(source_data, target_data, k = k))

    # MIDA
  } else if (method == "mida") {
    k     <- control$k     %||% 10
    sigma <- control$sigma %||% 1
    mu    <- control$mu    %||% 0.1
    max   <- control$max   %||% TRUE
    return(domain_adaptation_mida(source_data, target_data,
                                  k = k, sigma = sigma, mu = mu, max = max))

    # RD
  } else if (method == "rd") {
    return(domain_adaptation_riemannian(source_data, target_data))

    # CORAL
  } else if (method == "coral") {
    lambda <- control$lambda %||% 1e-5
    return(domain_adaptation_coral(source_data, target_data, lambda))

    # GFK
  } else if (method == "gfk") {
    dim_sub <- control$dim_subspace %||% 10
    return(domain_adaptation_gfk(source_data, target_data, dim_sub))

    # ART
  } else if (method == "art") {                  # two-argument ART routine
    return(domain_adaptation_art(source_data, target_data))

    # PT
  } else if (method == "pt") {                  # two-argument ART routine
    return(domain_adaptation_pt(source_data, target_data))

    # M3D
  } else if (method == "m3d") {

    if (is.null(control$source_labels))
      stop("method = 'm3d' requires  control$source_labels")

    stage1 <- control$stage1 %||%
      list(method = "tca",
           control = list(k = NULL, sigma = 1))

    stage2 <- control$stage2 %||%
      list(method = "sa",
           control = list(k = 10))

    return(domain_adaptation_m3d(
      source_data   = source_data,
      source_labels = control$source_labels,
      target_data   = target_data,
      stage1        = stage1,
      stage2        = stage2,
      l_iter        = control$l_iter        %||% 10,
      lambda_ridge  = control$lambda_ridge  %||% 1e-2,
      eta_kernel    = control$eta_kernel    %||% 0.1,
      label_offset  = control$label_offset  %||% 0,
      expl_var      = control$expl_var      %||% 0.90,
      max_dim       = control$max_dim       %||% 30))
  }
}


#' @importFrom stats  cov dist median prcomp
#' @importFrom transport  wasserstein1d
#' @importFrom geigen geigen
#' @importFrom graphics plot
#' @importFrom utils  head tail
NULL
