# MedSAM-TTA: Test-Time Adaptation for Medical Segmentation

This repository contains an implementation of Test-Time Adaptation (TTA) for the MedSAM model. It utilizes the Segment Anything Model (SAM) architecture and applies specific loss functions (DAL-CRF and Entropy Minimization) to adapt image embeddings during inference time for improved segmentation accuracy given bounding box prompts.

## Features
- **Feature Extraction**: Freezes SAM encoders and optimizes embeddings.
- **DAL-CRF Loss**: Distribution Alignment and Consistency Regularization.
- **Entropy Minimization**: Optimizes for high-confidence predictions.
- **Dynamic Learning Rate**: Adjusts based on foreground ratio estimation.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/MedSAM-TTA.git](https://github.com/yourusername/MedSAM-TTA.git)
   cd MedSAM-TTA
