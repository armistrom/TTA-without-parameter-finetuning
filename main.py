import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import MedSAM_TTA


def load_image(image_path, target_size=(1024, 1024)):
    img = plt.imread(image_path)[:, :, :3]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    img_tensor = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=False)
    return img_tensor


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    # Initialize model
    print(f"Loading model from {args.checkpoint}...")
    model = MedSAM_TTA(
        sam_checkpoint=args.checkpoint,
        model_type=args.model_type,
        k=args.k,
        iterations=args.iterations,
        lr=args.lr
    )

    # Load and preprocess image
    image = load_image(args.image_path)
    image = image.to(device)

    coords = list(map(int, args.box.split(',')))
    box = torch.tensor([[coords[0], coords[1]], [coords[2], coords[3]]], dtype=torch.int32)

    masks, ious, output_mask = model(
        image,
        box=box
    )

    print(f"Output masks shape: {masks.shape}")
    print(f"IOU predictions: {ious}")

    if args.output:
        torch.save(output_mask, args.output)
        print(f"Result saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedSAM Test-Time Adaptation (TTA)")

    parser.add_argument('--checkpoint', type=str, required=True, help='Path to SAM/MedSAM checkpoint (.pth)')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--box', type=str, default="144,160,826,631", help='Bounding box coordinates: x1,y1,x2,y2')
    parser.add_argument('--model_type', type=str, default="vit_b", help='SAM model type (vit_b, vit_l, vit_h)')
    parser.add_argument('--iterations', type=int, default=1, help='Number of TTA iterations')
    parser.add_argument('--k', type=int, default=4, help='Number of random samples for CRF')
    parser.add_argument('--lr', type=float, default=5e2, help='Learning rate')
    parser.add_argument('--output', type=str, default="output_mask.pt", help='Path to save output tensor')

    args = parser.parse_args()
    main(args)
