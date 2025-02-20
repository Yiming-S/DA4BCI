% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{rbf_kernel}
\alias{rbf_kernel}
\title{RBF (Gaussian) Kernel Computation}
\usage{
rbf_kernel(x, y, sigma)
}
\arguments{
\item{x}{A numeric matrix where rows correspond to observations and columns
to features.}

\item{y}{A numeric matrix with the same number of columns as \code{x}.}

\item{sigma}{A positive scalar for the RBF kernel bandwidth.}
}
\value{
A numeric matrix of size \eqn{nrow(x) \times nrow(y)} containing
the RBF kernel values.
}
\description{
The \code{rbf_kernel} function calculates the Radial Basis Function (RBF)
kernel matrix between two sets of observations \code{x} and \code{y},
using \eqn{\exp(-||x - y||^2 / \sigma^2)}. It returns only the cross-block
corresponding to \code{x} vs. \code{y}.
}
\details{
This function computes pairwise distances among rows of \code{x} and \code{y}
(stacked together) via \code{\link[stats]{dist}}, then transforms them using
the Gaussian kernel formula. Only the \code{x} vs. \code{y} sub-block is returned,
producing an \eqn{n_x \times n_y} matrix.
}
\examples{
\dontrun{
set.seed(123)
x <- matrix(rnorm(20), nrow = 5, ncol = 4)
y <- matrix(rnorm(24, mean = 2), nrow = 6, ncol = 4)
Kxy <- rbf_kernel(x, y, sigma = 1)
dim(Kxy)  # 5 x 6
}

}
