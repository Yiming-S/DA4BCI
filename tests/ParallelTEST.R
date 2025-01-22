
# install.packages("remotes")
remotes::install_github("Yiming-S/DA4BCI")
# ---------------------------------------------
# Parallel Domain Adaptation Testing Script
# ---------------------------------------------
library(DA4BCI)
ls("package:DA4BCI")

# Required Libraries
library(parallel)
library(doParallel)
library(foreach)
library(ggplot2)
library(gridExtra)
library(Rtsne)
library(RSpectra)
library(geigen)
library(MASS)

# Path for Saving Results
pic_dir <- "~/Desktop/pic"  # Adjust as needed

# Domain Adaptation Methods
DA_methods <- c("tca", "sa", "mida", "rd", "coral", "gfk")

# ---------------------------------------------
# (1) Generate Data
# ---------------------------------------------
generate_data <- function(n_s, n_t, dist_type, fs = 160, t = 3) {
  adj_n_s <- n_s * fs * t
  adj_n_t <- n_t * fs * t

  dist_name <- switch(
    as.character(dist_type),
    "1" = "Standard Normal Distribution",
    "2" = "Uniform Distribution",
    "3" = "Normal Distribution (Different Means)",
    "4" = "Exponential Distribution",
    "5" = "Normal Dist (Different SD)",
    "6" = "Poisson Distribution",
    "7" = "Student's t-Distribution",
    "8" = "Binomial Distribution",
    "9" = "Normal Dist (Another Variant)",
    "10" = "Normal + Cauchy Distribution",
    stop("Invalid dist_type.")
  )

  source_data <- switch(
    as.character(dist_type),
    "1"  = matrix(rnorm(adj_n_s * 50), adj_n_s, 50),
    "2"  = matrix(runif(adj_n_s * 50), adj_n_s, 50),
    "3"  = matrix(rnorm(adj_n_s * 50, mean = 5), adj_n_s, 50),
    "4"  = matrix(rexp(adj_n_s * 50), adj_n_s, 50),
    "5"  = matrix(rnorm(adj_n_s * 50, sd = 2), adj_n_s, 50),
    "6"  = matrix(rpois(adj_n_s * 50, lambda = 3), adj_n_s, 50),
    "7"  = matrix(rt(adj_n_s * 50, df = 5), adj_n_s, 50),
    "8"  = matrix(rbinom(adj_n_s * 50, size = 10, prob = 0.3), adj_n_s, 50),
    "9"  = matrix(rnorm(adj_n_s * 50), adj_n_s, 50),
    "10" = matrix(rnorm(adj_n_s * 50), adj_n_s, 50),
    stop("Invalid dist_type.")
  )

  target_data <- switch(
    as.character(dist_type),
    "1"  = matrix(rnorm(adj_n_t * 50), adj_n_t, 50),
    "2"  = matrix(runif(adj_n_t * 50), adj_n_t, 50),
    "3"  = matrix(rnorm(adj_n_t * 50, mean = -5), adj_n_t, 50),
    "4"  = matrix(rexp(adj_n_t * 50), adj_n_t, 50),
    "5"  = matrix(rnorm(adj_n_t * 50, sd = 0.5), adj_n_t, 50),
    "6"  = matrix(rpois(adj_n_t * 50, lambda = 10), adj_n_t, 50),
    "7"  = matrix(rt(adj_n_t * 50, df = 10), adj_n_t, 50),
    "8"  = matrix(rbinom(adj_n_t * 50, size = 10, prob = 0.7), adj_n_t, 50),
    "9"  = matrix(rnorm(adj_n_t * 50, sd = 3), adj_n_t, 50),
    "10" = matrix(rcauchy(adj_n_t * 50), adj_n_t, 50),
    stop("Invalid dist_type.")
  )

  list(source_data = source_data, target_data = target_data, dist_name = dist_name)
}

# ---------------------------------------------
# (2) Parallel Testing
# ---------------------------------------------
# Initialize Parallel Cluster
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Collect Results
all_results <- list()

for (method_name in DA_methods) {
  cat(">>> Testing method:", method_name, "\n")

  results_this_method <- foreach(
    i = 1:10,
    .packages = c("ggplot2", "gridExtra", "Rtsne", "RSpectra", "geigen", "MASS", "DA4BCI")
  ) %dopar% {
    # Generate test data
    test_data <- generate_data(10, 10, dist_type = i, fs = 50, t = 10)
    src <- test_data$source_data
    tgt <- test_data$target_data

    # Calculate MMD Before Adaptation
    mmd_before <- compute_mmd(src, tgt, sigma = 1)

    # Perform domain adaptation
    start_time <- Sys.time()
    da <- switch(
      method_name,
      tca = domain_adaptation_tca(src, tgt, k = 10, mu = 1e-5, sigma = 10),
      sa  = domain_adaptation_sa(src, tgt, k = 10),
      mida= domain_adaptation_mida(src, tgt, k = 10, max = TRUE),
      rd  = {
        rd_res <- domain_adaptation_riemannian(src, tgt)
        list(weighted_source_data = src %*% rd_res$rotation_matrix, target_data = tgt)
      },
      coral = domain_adaptation_coral(src, tgt, lambda = 1e-5),
      gfk   = domain_adaptation_gfk(src, tgt, dim_subspace = 10)
    )
    elapsed <- Sys.time() - start_time

    # Visualization
    Z_s <- da$weighted_source_data
    Z_t <- da$target_data

    # Calculate MMD After Adaptation
    mmd_after <- compute_mmd(Z_s, Z_t, sigma = 1)

    pdf(NULL)
    plots <- plot_data_comparison(src, tgt, Z_s, Z_t, description = test_data$dist_name)
    combined_plot <- grid.arrange(plots$p1, plots$p2, ncol = 2,
                                  top = paste("Method:", method_name, "| DistType:", i))
    dev.off()

    # Save plot
    ggsave(file.path(pic_dir, paste0(method_name, "_dist_", i, ".png")),
           combined_plot, width = 14, height = 7)

    # Return results for this distribution
    data.frame(
      Method = method_name, DistType = i,
      MMD_Before = mmd_before, MMD_After = mmd_after,
      Time_Taken = as.numeric(elapsed), stringsAsFactors = FALSE
    )
  }
  summary_table <- rbind(summary_table, do.call(rbind, results_this_method))
}

stopCluster(cl)

# ---------------------------------------------
# (3) Summarize Results
# ---------------------------------------------
for (method in names(all_results)) {
  times <- sapply(all_results[[method]], function(x) x$time_taken)
  cat("Method:", method, "| Avg runtime:", round(mean(times), 3), "seconds\n")
}

