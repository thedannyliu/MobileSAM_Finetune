import torch
from torch.utils.data import Dataset
import numpy as np
import random
from pathlib import Path
from PIL import Image
import os # Not strictly necessary with pathlib, but often present

class ComponentDataset(Dataset):
    """
    ComponentDataset for loading images and multiple object masks for finetuning SAM.
    Each image can have multiple associated object masks, and each image-mask pair
    will be treated as a distinct sample.
    Prompts (box or points) are generated based on the selected mask and mode.
    """
    def __init__(self, root_dir, transform=None, max_bbox_shift=10,
                 prompt_mode='mixed', min_points=1, max_points=3, image_size=1024):
        """
        Args:
            root_dir (string): Directory containing 'image' and 'mask' subdirectories.
                               - 'image' contains image files (e.g., xxx.jpg)
                               - 'mask' contains subdirectories named after image stems (e.g., xxx/)
                                 which in turn contain mask files (e.g., yyy-1.png).
            transform (tuple, optional): A tuple of two optional transforms to be applied
                on an image and its mask respectively. transform[0] for image, transform[1] for mask.
            max_bbox_shift (int, optional): Max random perturbation for bounding box coordinates.
            prompt_mode (str, optional): 'box', 'point', or 'mixed'.
            min_points (int, optional): Minimum number of points for a point prompt.
            max_points (int, optional): Maximum number of points for a point prompt.
            image_size (int, optional): The target size to which images and masks are resized.
        """
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / 'image'
        self.mask_dir = self.root_dir / 'mask'

        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Image or mask directory not found in {root_dir}. "
                                    f"Expected subdirectories 'image' and 'mask'.")

        self.transform_image = transform[0] if transform and len(transform) > 0 else None
        self.transform_mask = transform[1] if transform and len(transform) > 1 else None
        
        self.max_bbox_shift = max_bbox_shift
        self.prompt_mode = prompt_mode
        self.min_points = min_points
        self.max_points = max_points
        self.image_size = image_size # Needed for bbox perturbation limits

        self.samples = []
        image_files = list(self.image_dir.glob('*.jpg')) + \
                      list(self.image_dir.glob('*.jpeg')) + \
                      list(self.image_dir.glob('*.png'))

        for img_path in image_files:
            img_stem = img_path.stem
            corresponding_mask_subdir = self.mask_dir / img_stem
            if corresponding_mask_subdir.is_dir():
                mask_files_in_subdir = sorted(list(corresponding_mask_subdir.glob('*.png')))
                if mask_files_in_subdir:
                    for mask_path in mask_files_in_subdir:
                        self.samples.append({'image': img_path, 'mask': mask_path})
                else:
                    print(f"Warning: No mask PNGs found in {corresponding_mask_subdir} for image {img_path.name}")
            # else:
                # Optional: print warning if no mask subdir for an image, can be noisy if many such images
                # print(f"Warning: No mask directory found at {corresponding_mask_subdir} for image {img_path.name}")


        if not self.samples:
            raise ValueError(f"No image-mask pairs found. Check dataset structure in {root_dir}.")
        
        print(f"Initialized ComponentDataset with {len(self.samples)} samples. Prompt mode: {self.prompt_mode}")


    def __len__(self):
        return len(self.samples)

    def compute_bbox(self, mask_tensor):
        """
        Compute the bounding box of the foreground region in a binary mask tensor.
        Args:
            mask_tensor (tensor): A binary mask tensor (C, H, W) or (H, W). Assumes values are 0 or 1 (or >0 for foreground).
        Returns:
            tensor: A tensor containing coordinates (x_min, y_min, x_max, y_max) of the bbox.
        """
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.squeeze(0) # Reduce to (H, W)

        # Consider foreground where mask > 0 (robust to slight variations if not strictly 0 or 1)
        # For SAM finetuning, masks from ToTensor() are usually [0,1] float.
        # Using a small threshold like 0.5 or checking for >0 is common.
        # The original code used `mask_tensor == 1`. If your masks are strictly 0/1, that's fine.
        # Using `mask_tensor > 0` or `mask_tensor > 0.5` can be more general.
        thresholded_mask = mask_tensor > 0.5 

        rows_any_white = torch.any(thresholded_mask, dim=1)
        cols_any_white = torch.any(thresholded_mask, dim=0)

        rows_white_indices = torch.where(rows_any_white)[0]
        cols_white_indices = torch.where(cols_any_white)[0]

        if rows_white_indices.nelement() == 0 or cols_white_indices.nelement() == 0:
            # No foreground pixels, return a zero bbox (or handle as error/default)
            return torch.tensor([0, 0, 0, 0], dtype=torch.float)

        y_min, y_max = rows_white_indices[0].item(), rows_white_indices[-1].item()
        x_min, x_max = cols_white_indices[0].item(), cols_white_indices[-1].item()

        return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float)

    def compute_point_prompts(self, mask_tensor):
        """
        Generate random point prompts from the foreground of the mask.
        Args:
            mask_tensor (tensor): Binary mask tensor (C, H, W) or (H, W).
        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]:
                - point_coords_padded (max_points, 2): Padded (x,y) coordinates.
                - point_labels_padded (max_points,): Padded labels (1 for fg).
        """
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.squeeze(0) # Reduce to (H, W)

        # Similar to compute_bbox, ensure we correctly identify foreground.
        foreground_pixels_yx = torch.argwhere(mask_tensor > 0.5) # Get (row, col) i.e., (y, x)

        point_coords_padded = torch.zeros((self.max_points, 2), dtype=torch.float)
        point_labels_padded = torch.zeros(self.max_points, dtype=torch.long) # SAM expects long/int labels

        if foreground_pixels_yx.shape[0] == 0:
            return point_coords_padded, point_labels_padded # No points to sample

        num_points_to_sample = random.randint(self.min_points, self.max_points)
        num_actual_points = min(num_points_to_sample, foreground_pixels_yx.shape[0])

        if num_actual_points > 0:
            indices = torch.randperm(foreground_pixels_yx.shape[0])[:num_actual_points]
            sampled_points_yx = foreground_pixels_yx[indices] # Shape (num_actual_points, 2), order (y,x)
            
            # SAM expects (x,y) coordinates
            sampled_points_xy = torch.flip(sampled_points_yx, dims=[1]).float() # Convert to (x,y) and float
            
            point_coords_padded[:num_actual_points] = sampled_points_xy
            point_labels_padded[:num_actual_points] = 1 # Foreground points

        return point_coords_padded, point_labels_padded


    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        img_path = sample_info['image']
        mask_path = sample_info['mask']

        image = Image.open(img_path).convert("RGB")
        # Ensure mask is loaded as single channel (grayscale)
        # PIL's "L" mode is 8-bit pixels, black and white.
        mask = Image.open(mask_path).convert("L") 

        if self.transform_image:
            transformed_image = self.transform_image(image)
        else: # Fallback if no transform provided (e.g. for debugging)
            transformed_image = transforms.ToTensor()(image)


        if self.transform_mask:
            transformed_mask = self.transform_mask(mask)
        else: # Fallback
            transformed_mask = transforms.ToTensor()(mask)
        
        # Ensure mask is binary [0,1] after transform for robust computations
        # ToTensor typically scales L mode images (0-255) to [0,1] float.
        # If masks are guaranteed to be 0 or 255, this works fine.
        # Otherwise, explicit binarization might be needed.
        # e.g. transformed_mask = (transformed_mask > 0.5).float()
        # For now, assuming ToTensor() and mask content is sufficient.

        # Initialize prompts
        box_prompt_coords = torch.zeros(4, dtype=torch.float) # (x_min, y_min, x_max, y_max)
        point_coords_padded = torch.zeros((self.max_points, 2), dtype=torch.float) # (N_max, 2) for (x,y)
        point_labels_padded = torch.zeros(self.max_points, dtype=torch.long) # (N_max,)
        
        is_point_prompt_item = False # Flag to indicate if this item uses point prompt

        # Determine current prompt type for this item
        current_item_prompt_type = self.prompt_mode
        if self.prompt_mode == 'mixed':
            current_item_prompt_type = random.choice(['box', 'point'])

        if current_item_prompt_type == 'box':
            raw_bbox = self.compute_bbox(transformed_mask) # x_min, y_min, x_max, y_max
            
            # Apply perturbation if a valid bbox was found
            if not torch.all(raw_bbox == 0): # Check if bbox is not all zeros
                # Perturbation logic (ensure coordinates are within image bounds after shift)
                # The image_size is the size of transformed_image and transformed_mask (H, W)
                # Assuming transformed_mask is (C, H, W) or (H, W)
                h, w = transformed_mask.shape[-2], transformed_mask.shape[-1]
                
                bbox_width = raw_bbox[2] - raw_bbox[0]
                bbox_height = raw_bbox[3] - raw_bbox[1]

                noise_w_val = 0
                noise_h_val = 0

                if bbox_width.item() > 0 and self.max_bbox_shift > 0: # Ensure positive width
                    noise_w_val = torch.clamp(
                        torch.randn(1) * bbox_width * 0.1, # Scale noise by 10% of bbox dim
                        min=-self.max_bbox_shift, max=self.max_bbox_shift
                    ).round().int().item()
                if bbox_height.item() > 0 and self.max_bbox_shift > 0: # Ensure positive height
                    noise_h_val = torch.clamp(
                        torch.randn(1) * bbox_height * 0.1,
                        min=-self.max_bbox_shift, max=self.max_bbox_shift
                    ).round().int().item()
                
                x_min_shifted = max(0, raw_bbox[0].item() + noise_w_val)
                y_min_shifted = max(0, raw_bbox[1].item() + noise_h_val)
                x_max_shifted = min(w, raw_bbox[2].item() + noise_w_val) # w is width of image
                y_max_shifted = min(h, raw_bbox[3].item() + noise_h_val) # h is height of image

                # Ensure x_max > x_min and y_max > y_min after shift
                if x_max_shifted <= x_min_shifted: x_max_shifted = x_min_shifted + 1 # Ensure min width of 1
                if y_max_shifted <= y_min_shifted: y_max_shifted = y_min_shifted + 1 # Ensure min height of 1
                x_max_shifted = min(w, x_max_shifted) # Re-clamp if +1 pushed it over
                y_max_shifted = min(h, y_max_shifted)


                box_prompt_coords = torch.tensor([
                    x_min_shifted, y_min_shifted, x_max_shifted, y_max_shifted
                ], dtype=torch.float)
            else: # if no object in mask or bbox calculation failed, use the zero box.
                box_prompt_coords = raw_bbox # which is torch.zeros(4)
            
            is_point_prompt_item = False

        elif current_item_prompt_type == 'point':
            # compute_point_prompts already handles padding to self.max_points
            point_coords_padded, point_labels_padded = self.compute_point_prompts(transformed_mask)
            is_point_prompt_item = True
            # box_prompt_coords remains zeros for point prompts

        return transformed_image, transformed_mask, box_prompt_coords, point_coords_padded, point_labels_padded, is_point_prompt_item