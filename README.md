# DA4BCI

**DA4BCI** is an R package that provides multiple domain adaptation methods tailored for EEG-based brain-computer interface (BCI) applications. It includes a unified interface for:

- **TCA (Transfer Component Analysis)**
- **SA (Subspace Alignment)**
- **MIDA (Maximum Independence Domain Adaptation)**
- **RD (Riemannian Distance)**
- **CORAL (Correlation Alignment)**
- **GFK (Geodesic Flow Kernel)**

These methods help align EEG data from different sessions or subjects, mitigating distributional shifts and enabling more robust learning.

## Installation

1. Make sure you have R 3.5.0 or later.
2. In R, install the **remotes** (or **devtools**) package if you haven’t yet:
   ```r
   install.packages("remotes")
   remotes::install_github("Yiming-S/DA4BCI", force = TRUE)
