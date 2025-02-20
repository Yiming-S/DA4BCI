% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{compute_wasserstein}
\alias{compute_wasserstein}
\title{Compute Wasserstein Distance}
\usage{
compute_wasserstein(source, target)
}
\arguments{
\item{source}{A numeric matrix where rows are observations and columns are features.}

\item{target}{A numeric matrix with the same structure as \code{source}.}
}
\value{
A single numeric value representing the Wasserstein distance.
}
\description{
The \code{compute_wasserstein} function calculates the Wasserstein distance
between the distributions represented by \code{source} and \code{target}.
}
\details{
This function uses the \code{transport} package to compute the optimal transport plan
and calculate the Wasserstein distance. Pairwise Euclidean distances are used as the cost matrix.
}
\examples{
\dontrun{
source <- matrix(rnorm(20), nrow = 5, ncol = 4)
target <- matrix(rnorm(24, mean = 2), nrow = 6, ncol = 4)
wasserstein_dist <- compute_wasserstein(source, target)
cat("Wasserstein Distance:", wasserstein_dist, "\n")
}

}
