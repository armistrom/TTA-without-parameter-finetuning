# Test-time Adaptation for Foundation Medical Segmentation Model without Parametric Updates

This repository contains an implementation of the paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Test-time_Adaptation_for_Foundation_Medical_Segmentation_Model_Without_Parametric_Updates_ICCV_2025_paper.pdf. 

This implementation uses a pretrained MEDSAM model with a vit_b backbone imported from segment_anything.

## Features
- **Feature Extraction**: Freezes SAM encoders and optimises embeddings.
- **DAL-CRF Loss**: Distribution Alignment and Consistency Regularisation.
- **Entropy Minimization**: Optimises for high-confidence predictions and ensures the latent refinement aligns with the decoder's distribution.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/armistrom/TTA-without-parameter-finetuning.git
   
