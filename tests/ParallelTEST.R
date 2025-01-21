
# install.packages("remotes")
remotes::install_github("Yiming-S/DA4BCI")
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
  adjusted_n_s <- n_s * fs * t
  adjusted_n_t <- n_t * fs * t

  if (dist_type == 1) {
    source_data <- matrix(rnorm(adjusted_n_s * 50), adjusted_n_s, 50)
    target_data <- matrix(rnorm(adjusted_n_t * 50), adjusted_n_t, 50)
    dist_name <- "Standard Normal Distribution"
  } else if (dist_type == 2) {
    source_data <- matrix(runif(adjusted_n_s * 50), adjusted_n_s, 50)
    target_data <- matrix(runif(adjusted_n_t * 50), adjusted_n_t, 50)
    dist_name <- "Uniform Distribution"
  } else if (dist_type == 3) {
    source_data <- matrix(rnorm(adjusted_n_s * 50, mean = 5), adjusted_n_s, 50)
    target_data <- matrix(rnorm(adjusted_n_t * 50, mean = -5), adjusted_n_t, 50)
    dist_name <- "Normal Distribution with Different Means"
  } else if (dist_type == 4) {
    source_data <- matrix(rexp(adjusted_n_s * 50), adjusted_n_s, 50)
    target_data <- matrix(rexp(adjusted_n_t * 50), adjusted_n_t, 50)
    dist_name <- "Exponential Distribution"
  } else if (dist_type == 5) {
    source_data <- matrix(rnorm(adjusted_n_s * 50, sd = 2), adjusted_n_s, 50)
    target_data <- matrix(rnorm(adjusted_n_t * 50, sd = 0.5), adjusted_n_t, 50)
    dist_name <- "Normal Dist with Diff. Standard Deviations"
  } else if (dist_type == 6) {
    source_data <- matrix(rpois(adjusted_n_s * 50, lambda = 3), adjusted_n_s, 50)
    target_data <- matrix(rpois(adjusted_n_t * 50, lambda = 10), adjusted_n_t, 50)
    dist_name <- "Poisson Distribution"
  } else if (dist_type == 7) {
    source_data <- matrix(rt(adjusted_n_s * 50, df = 5), adjusted_n_s, 50)
    target_data <- matrix(rt(adjusted_n_t * 50, df = 10), adjusted_n_t, 50)
    dist_name <- "Student's t-Distribution"
  } else if (dist_type == 8) {
    source_data <- matrix(rbinom(adjusted_n_s * 50, size = 10, prob = 0.3),
                          adjusted_n_s, 50)
    target_data <- matrix(rbinom(adjusted_n_t * 50, size = 10, prob = 0.7),
                          adjusted_n_t, 50)
    dist_name <- "Binomial Distribution"
  } else if (dist_type == 9) {
    source_data <- matrix(rnorm(adjusted_n_s * 50, mean = 0, sd = 1),
                          adjusted_n_s, 50)
    target_data <- matrix(rnorm(adjusted_n_t * 50, mean = 0, sd = 3),
                          adjusted_n_t, 50)
    dist_name <- "Normal Dist with Diff. SD (another variant)"
  } else if (dist_type == 10) {
    source_data <- matrix(rnorm(adjusted_n_s * 50), adjusted_n_s, 50)
    target_data <- matrix(rcauchy(adjusted_n_t * 50), adjusted_n_t, 50)
    dist_name <- "Normal and Cauchy Distribution"
  }

  # Generate random labels (0 or 1) for each row
  source_label <- sample(0:1, adjusted_n_s, replace = TRUE)
  target_label <- sample(0:1, adjusted_n_t, replace = TRUE)

  return(list(source_data = source_data, target_data = target_data,
              source_label = source_label, target_label = target_label,
              dist_name = dist_name))
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
                                                                # cov_s <- cov(source_data)
                                                                # cov_t <- cov(target_data)
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
