% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{orthonormal_complement}
\alias{orthonormal_complement}
\title{Orthonormal Complement of a Matrix}
\usage{
orthonormal_complement(U)
}
\arguments{
\item{U}{A \eqn{d \times k} matrix with orthonormal columns.}
}
\value{
A \eqn{d \times (d-k)} matrix (or fewer columns if some are removed),
with orthonormal columns that are orthogonal to every column of \code{U}.
}
\description{
The \code{orthonormal_complement} function computes a matrix whose columns form
an orthonormal basis for the subspace orthogonal to \code{U} in \eqn{\mathbb{R}^d}.
Useful in methods like GFK, where one needs both the principal subspace and
its complement.
}
\details{
The function uses a QR decomposition of the \eqn{d \times d} identity matrix
to get a full orthonormal basis, then projects it onto the orthogonal complement
of \code{U}. Columns close to zero (norm < 1e-8) are removed. The result is truncated
to \eqn{d-k} columns if needed.
}
\examples{
\dontrun{
set.seed(123)
U <- qr.Q(qr(matrix(rnorm(12), nrow = 4, ncol = 3)))  # 4x3 orthonormal basis
U_perp <- orthonormal_complement(U)
dim(U_perp)  # should be 4 x 1
# Check orthogonality: crossprod(U, U_perp) ~ 0
}

}
