import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from itertools import combinations
import numpy as np
import random
import utils


class MedSAM_TTA(nn.Module):
    def __init__(self, sam_checkpoint, model_type="vit_b", k=4, iterations=1, lr=1e-4, lambda_weight=1e3):
        super().__init__()

        # Load SAM model
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sam.to(self.device)

        self.k = k
        self.iterations = iterations
        self.lambda_weight = lambda_weight
        self.learning_rate = lr

        # Freeze all components using utility function
        utils.freeze_components(self.sam)

    def extract_features(self, image):
        """
        Extract features from the image encoder.
        Args:
            image: Preprocessed image tensor [B, 3, 1024, 1024]
        """
        self.image = image
        print(f'Processing image shape: {image.shape}')
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(image)
        print(f'Embedding shape: {image_embeddings.shape}')
        return image_embeddings

    def receptive_field(self, row, column, kernel=16):
        return torch.tensor(self.image[:, :, row:row + kernel, column:column + kernel].flatten())

    def random_sample(self, box):
        coords = [(x, y) for x in range(64) for y in range(64)]
        coords_arr = np.array(coords).reshape(64, 64, 2)

        if box is None:
            cords = coords_arr.reshape(-1, 2).tolist()
            random.seed(23)
            sampled_coords = random.sample(cords, self.k)
        else:
            # Rejecting all the coordinates outside the bounding box region
            r1, c1 = box[0]
            r2, c2 = box[1]

            # Coordinate scaling for embeddings
            r1, r2 = sorted([np.floor(r1 / 16), np.floor(r2 / 16)])
            c1, c2 = sorted([np.floor(c1 / 16), np.floor(c2 / 16)])
            r1, r2, c1, c2 = int(r1), int(r2), int(c1), int(c2)

            cords = coords_arr[r1:r2, c1:c2].reshape(-1, 2).tolist()
            random.seed(23)
            sampled_coords = random.sample(cords, self.k)

        pairs = list(combinations(sampled_coords, 2))
        self.combinations = torch.tensor(pairs, dtype=torch.int32)

    def dal_crf(self):
        loss = 0
        for i in self.combinations:
            r1, c1 = i[0, 0], i[0, 1]
            r2, c2 = i[1, 0], i[1, 1]

            embed1 = self.image_embed[:, :, r1, c1]
            embed2 = self.image_embed[:, :, r2, c2]

            rf1 = self.receptive_field(r1, c1)
            rf2 = self.receptive_field(r2, c2)

            MMD_rf = utils.calculate_mmd(rf1, rf2)  # MMD from utils
            fro_norm = torch.norm(embed1 - embed2, p='fro')

            loss += ((1 / 2) * (MMD_rf) * (torch.exp(fro_norm)))

        loss_dal_crf = loss / self.k
        print(f'Loss DAL-CRF: {loss_dal_crf}')
        return loss_dal_crf

    def loss_calc(self, mask_logits):
        loss_dal_crf = self.dal_crf()
        loss_em = utils.calculate_entropy(mask_logits)
        print(f'Loss EM: {loss_em.sum()}')

        final_loss = loss_dal_crf.sum() + self.lambda_weight * loss_em.sum()
        print(f'Final_Loss: {final_loss}')
        return final_loss

    def params_pred(self, masks):
        """Dynamic Learning Rate Adjustment based on foreground ratio"""
        probs = torch.sigmoid(masks)
        high_conf_mask = probs > 0.95
        high_conf_mask_int = high_conf_mask.int()

        num_foreground = high_conf_mask_int.sum().item()
        total_pixels = high_conf_mask.numel()
        fe = float(num_foreground) / total_pixels

        # lr interpolation based on foreground ratio
        low, high = 0.05, 0.2
        low_lr, high_lr = 1e-1, 5e2

        if fe < low:
            te = low_lr
        elif fe > high:
            te = high_lr
        else:
            t = (fe - low) / (high - low)
            te = low_lr + t * (high_lr - low_lr)

        self.learning_rate = torch.tensor(te)

    def predict_masks(self, image_embeddings, point_coords=None, point_labels=None,
                      box=None, mask_input=None, multimask_output=True,
                      return_logits=True):
        """
        Generate masks using manipulated features and prompt encoder.
        """
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=(point_coords, point_labels) if point_coords is not None else None,
                boxes=box,
                masks=mask_input,
            )

            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            masks = self.sam.postprocess_masks(
                low_res_masks,
                input_size=(1024, 1024),
                original_size=(1024, 1024)
            )

            if not return_logits:
                masks = (masks > 0.0).float()

        return masks, iou_predictions, low_res_masks

    def forward(self, image, box=None, multimask_output=True):
        """
        Full forward pass: extract features -> manipulate -> predict masks
        """
        # Extract features
        image_embeddings = self.extract_features(image)
        self.random_sample(box)

        box_flat = box.flatten().unsqueeze(0).to(torch.float32)

        # Create learnable parameter from embeddings
        image_embeddings = torch.nn.Parameter(image_embeddings.detach().clone(), requires_grad=True)
        optimizer = torch.optim.Adam([image_embeddings], lr=self.learning_rate)
        self.image_embed = image_embeddings

        # Initial Prediction
        masks, iou_predictions, _ = self.predict_masks(
            image_embeddings,
            multimask_output=multimask_output,
            box=box_flat
        )

        # Optimization Loop
        for i in range(self.iterations):
            self.params_pred(masks)

            optimizer.zero_grad()
            final_loss = self.loss_calc(masks)
            final_loss.backward()
            optimizer.step()

            masks, iou_predictions, _ = self.predict_masks(
                image_embeddings,
                multimask_output=multimask_output,
                box=box_flat
            )
            self.random_sample(box)

        # Final Output Processing
        image_embeddings = image_embeddings.detach()
        probs = torch.sigmoid(masks)
        high_conf_mask = probs > 0.95
        return masks, iou_predictions, high_conf_mask.int()