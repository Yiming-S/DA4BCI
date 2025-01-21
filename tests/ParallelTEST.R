####################################
# Example Parallel Domain Adaptation Script
####################################

# Required Packages
library(parallel)
library(doParallel)
library(foreach)
library(ggplot2)
library(gridExtra)
library(Rtsne)
library(RSpectra)
library(geigen)
library(MASS)

####################################
# (1) Generate Various Distributions
####################################
generate_data <- function(n_s, n_t, dist_type, fs = 50, t = 1) {
  # Adjust the number of samples based on frequency (fs) and time (t)
  adj_n_s <- n_s * fs * t
  adj_n_t <- n_t * fs * t

  # A descriptive name for each distribution type
  dist_name <- switch(
    as.character(dist_type),
    "1"  = "Standard Normal Distribution",
    "2"  = "Uniform Distribution",
    "3"  = "Normal Dist with Different Means",
    "4"  = "Exponential Distribution",
    "5"  = "Normal Dist with Different SD",
    "6"  = "Poisson Distribution",
    "7"  = "Student's t-Distribution",
    "8"  = "Binomial Distribution",
    "9"  = "Normal Dist (variant) with Different SD",
    "10" = "Normal and Cauchy Distribution",
    stop("Invalid dist_type. Must be between 1 and 10.")
  )

  # Generate source/target data according to dist_type
  if (dist_type == 1) {
    source_data <- matrix(rnorm(adj_n_s * 50), adj_n_s, 50)
    target_data <- matrix(rnorm(adj_n_t * 50), adj_n_t, 50)
  } else if (dist_type == 2) {
    source_data <- matrix(runif(adj_n_s * 50), adj_n_s, 50)
    target_data <- matrix(runif(adj_n_t * 50), adj_n_t, 50)
  } else if (dist_type == 3) {
    source_data <- matrix(rnorm(adj_n_s * 50, mean = 5), adj_n_s, 50)
    target_data <- matrix(rnorm(adj_n_t * 50, mean = -5), adj_n_t, 50)
  } else if (dist_type == 4) {
    source_data <- matrix(rexp(adj_n_s * 50), adj_n_s, 50)
    target_data <- matrix(rexp(adj_n_t * 50), adj_n_t, 50)
  } else if (dist_type == 5) {
    source_data <- matrix(rnorm(adj_n_s * 50, sd = 2), adj_n_s, 50)
    target_data <- matrix(rnorm(adj_n_t * 50, sd = 0.5), adj_n_t, 50)
  } else if (dist_type == 6) {
    source_data <- matrix(rpois(adj_n_s * 50, lambda = 3), adj_n_s, 50)
    target_data <- matrix(rpois(adj_n_t * 50, lambda = 10), adj_n_t, 50)
  } else if (dist_type == 7) {
    source_data <- matrix(rt(adj_n_s * 50, df = 5), adj_n_s, 50)
    target_data <- matrix(rt(adj_n_t * 50, df = 10), adj_n_t, 50)
  } else if (dist_type == 8) {
    source_data <- matrix(rbinom(adj_n_s * 50, size = 10, prob = 0.3), adj_n_s, 50)
    target_data <- matrix(rbinom(adj_n_t * 50, size = 10, prob = 0.7), adj_n_t, 50)
  } else if (dist_type == 9) {
    source_data <- matrix(rnorm(adj_n_s * 50), adj_n_s, 50)
    target_data <- matrix(rnorm(adj_n_t * 50, sd = 3), adj_n_t, 50)
  } else {
    source_data <- matrix(rnorm(adj_n_s * 50), adj_n_s, 50)
    target_data <- matrix(rcauchy(adj_n_t * 50), adj_n_t, 50)
  }

  # Optional: random labels
  source_label <- sample(0:1, adj_n_s, replace = TRUE)
  target_label <- sample(0:1, adj_n_t, replace = TRUE)

  list(
    source_data = source_data,
    target_data = target_data,
    source_label = source_label,
    target_label = target_label,
    dist_name = dist_name
  )
}

####################################
# (2) Placeholder Domain Adaptation Functions
####################################
# Please replace these with your actual implementations
domain_adaptation_tca <- function(src, tgt, k=10, mu=1, sigma=1) {
  list(weighted_source_data=src, target_data=tgt)
}
domain_adaptation_sa <- function(src, tgt, k=10) {
  list(weighted_source_data=src, target_data=tgt)
}
domain_adaptation_mida <- function(src, tgt, k=10, max=TRUE) {
  list(weighted_source_data=src, target_data=tgt)
}
domain_adaptation_riemannian <- function(cov_s, cov_t) {
  # Return a dummy rotation
  list(rotation_matrix=diag(nrow(cov_s)))
}
domain_adaptation_coral <- function(src, tgt, lambda=1e-5) {
  list(weighted_source_data=src, target_data=tgt)
}
domain_adaptation_gfk <- function(src, tgt, dim_subspace=10) {
  list(weighted_source_data=src, target_data=tgt)
}

####################################
# (3) Simple Visualization
####################################
plot_data_comparison <- function(src, tgt, Zs, Zt, description, method="pca") {
  p1 <- ggplot() + ggtitle(paste("Before -", description))
  p2 <- ggplot() + ggtitle(paste("After -", description))
  list(p1 = p1, p2 = p2)
}

####################################
# (4) Main Parallel Testing Script
####################################
# Adjust to your desired output path
pic_dir <- "~/Desktop/pic"

# Domain adaptation methods to test
DA_methods <- c("tca", "sa", "mida", "rd", "coral", "gfk")

# Create and register a parallel cluster
num_cores <- max(1, detectCores() - 1)
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Collect results in a list
all_results <- list()

for (method_name in DA_methods) {
  cat(">>> Testing method:", method_name, "\n")

  # Parallel loop over 10 distribution types
  results_this_method <- foreach(
    i = 1:10,
    .packages = c("ggplot2", "gridExtra", "Rtsne","RSpectra","geigen","MASS")
  ) %dopar% {

    # (a) Generate data
    test_data <- generate_data(10, 10, dist_type = i, fs = 50, t = 3)
    src <- test_data$source_data
    tgt <- test_data$target_data

    # (b) Time the domain adaptation
    start_time <- Sys.time()
    da <- switch(
      method_name,
      tca = domain_adaptation_tca(src, tgt, k=10, mu=1e-5, sigma=10),
      sa  = domain_adaptation_sa(src, tgt, k=10),
      mida= domain_adaptation_mida(src, tgt, k=10, max=TRUE),
      rd  = {
        cov_s <- cov(src)
        cov_t <- cov(tgt)
        rd_res <- domain_adaptation_riemannian(cov_s, cov_t)
        list(weighted_source_data=src %*% rd_res$rotation_matrix, target_data=tgt)
      },
      coral = domain_adaptation_coral(src, tgt, lambda=1e-5),
      gfk   = domain_adaptation_gfk(src, tgt, dim_subspace=10)
    )
    elapsed <- Sys.time() - start_time

    # (c) Visualization
    Z_s <- da$weighted_source_data
    Z_t <- da$target_data

    plots <- plot_data_comparison(src, tgt, Z_s, Z_t,
                                  description = test_data$dist_name,
                                  method="pca")
    title_str <- paste0("Method:", method_name,
                        " | DistType:", i,
                        " | Time:", round(as.numeric(elapsed), 3), "s")
    combined_plot <- grid.arrange(plots$p1, plots$p2, ncol=2, top=title_str)

    # (d) Save figure
    out_file <- file.path(pic_dir, paste0(method_name, "_dist_", i, ".png"))
    ggsave(out_file, combined_plot, width=14, height=7)

    list(dist_type = i, time_taken = elapsed)
  }

  all_results[[method_name]] <- results_this_method
}

# Stop parallel cluster
stopCluster(cl)

# Summarize average runtime per method
for (m in names(all_results)) {
  times_vec <- sapply(all_results[[m]], function(x) as.numeric(x$time_taken))
  cat("Method:", m, "| Avg time:", round(mean(times_vec), 3), "seconds\n")
}
