
devtools::document()
install.packages()
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
pic_dir <- "~/Desktop/pic"


DA_methods <- c("tca", "sa", "mida", "rd", "coral", "gfk")


cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)


all_results <- list()

for(method_name in DA_methods) {
  cat(">>> Testing method:", method_name, "\n")


  results_this_method <- foreach(i = 1:10,
                                 .packages = c("ggplot2", "gridExtra",
                                               "Rtsne","RSpectra","geigen","MASS", "DA4BCI")) %dopar% {
                                                 test_data <- generate_data(10, 10, dist_type = i, fs = 50, t = 3)
                                                 source_data <- test_data$source_data
                                                 target_data <- test_data$target_data


                                                 tm <- Sys.time()
                                                 da <- switch(method_name,
                                                              tca = domain_adaptation_tca(source_data, target_data,
                                                                                          k = 10, mu = 1e-5, sigma = 10),
                                                              sa = domain_adaptation_sa(source_data, target_data, k = 10),
                                                              mida = domain_adaptation_mida(source_data, target_data,
                                                                                            k = 10, max = TRUE),
                                                              rd = {
                                                                rd_res <- domain_adaptation_riemannian(source_data, target_data)
                                                                list(weighted_source_data = source_data %*% rd_res$rotation_matrix,
                                                                     target_data = target_data)
                                                              },
                                                              coral = {
                                                                domain_adaptation_coral(source_data, target_data, lambda = 1e-5)
                                                              },
                                                              gfk = {
                                                                domain_adaptation_gfk(source_data, target_data, dim_subspace = 10)
                                                              }
                                                 )
                                                 time_taken <- Sys.time() - tm

                                                 Z_s <- da$weighted_source_data
                                                 Z_t <- da$target_data

                                                 # 可视化
                                                 plots <- plot_data_comparison(source_data, target_data,
                                                                               Z_s, Z_t, description = test_data$dist_name,
                                                                               method = "pca")

                                                 title <- paste0("Method:", method_name,
                                                                 " | DistType:", i,
                                                                 " | Time: ", round(as.numeric(time_taken), 3), "s")
                                                 combined_plot <- grid.arrange(plots$p1, plots$p2, ncol = 2, top = title)

                                                 combined_file <- file.path(pic_dir, paste0(method_name,
                                                                                            "_data_distribution_", i, ".png"))
                                                 ggsave(combined_file, combined_plot, width = 14, height = 7)

                                                 return(list(dist_type = i, time_taken = time_taken))
                                               }

  all_results[[method_name]] <- results_this_method
}

stopCluster(cl)

for(m in names(all_results)) {
  times_vec <- sapply(all_results[[m]], function(x) as.numeric(x$time_taken))
  cat("Method:", m, " | Avg time:", round(mean(times_vec), 3), "seconds\n")
}
