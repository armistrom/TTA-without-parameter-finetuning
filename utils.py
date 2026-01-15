import torch
import numpy as np


def freeze_components(sam_model):
    """
    Freeze image encoder, prompt encoder, and mask decoder.
    """
    # Freeze image encoder
    for param in sam_model.image_encoder.parameters():
        param.requires_grad = False

    # Freeze prompt encoder
    for param in sam_model.prompt_encoder.parameters():
        param.requires_grad = False

    # Freeze mask decoder
    for param in sam_model.mask_decoder.parameters():
        param.requires_grad = False

    print("✓ Image Encoder frozen")
    print("✓ Prompt Encoder frozen")
    print("✓ Mask Decoder frozen")


def gaussian_kernel(x, y, sigma):
    """
    Computes Gaussian RBF kernel matrix between x and y.
    x: (N, 1)
    y: (M, 1)
    """
    x = x.unsqueeze(1)  # (N, 1, 1)
    y = y.unsqueeze(0)  # (1, M, 1)
    diff = x - y  # (N, M, 1)
    return torch.exp(-torch.sum(diff ** 2, dim=-1) / (2 * sigma ** 2))  # (N, M)


def calculate_mmd(x, y, sigma=1.0):
    """
    Maximum Mean Discrepancy calculation.
    """
    x = x.float().view(-1, 1)
    y = y.float().view(-1, 1)

    Kxx = gaussian_kernel(x, x, sigma)
    Kyy = gaussian_kernel(y, y, sigma)
    Kxy = gaussian_kernel(x, y, sigma)

    mmd2 = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return torch.sqrt(torch.relu(mmd2))


def calculate_entropy(mask_logits):
    """
    Calculate pixel-wise entropy from mask logits.
    """
    # Convert logits to probabilities
    probs = torch.sigmoid(mask_logits)
    # Create mask for high confidence pixels (prob > 0.95)
    high_conf_mask = probs > 0.95

    # Calculate binary entropy: -[p*log(p) + (1-p)*log(1-p)]
    # Add small epsilon to avoid log(0)
    eps = 1e-7
    probs_clipped = torch.clamp(probs, eps, 1 - eps)

    # Binary entropy formula
    entropy = -(probs_clipped * torch.log(probs_clipped) +
                (1 - probs_clipped) * torch.log(1 - probs_clipped))

    # Zero out entropy for pixels below threshold
    entropy = entropy * high_conf_mask.float()

    total_entropy = entropy.sum(dim=(-2, -1))
    return total_entropy