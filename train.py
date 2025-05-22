from finetune_utils.datasets import ComponentDataset
from torchvision import transforms as T
import argparse, json, pathlib, os, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mobile_sam import sam_model_registry
from finetune_utils.distill_losses import (
    encoder_matching_loss, decoder_matching_loss,
    attention_matching_loss, rkd_loss
)
from finetune_utils.feature_hooks import register_hooks, pop_features


def load_cached_npy_features(base_precomputed_dir: pathlib.Path, # e.g., ./precomputed specified in mobileSAM.json
                             teacher_name: str,                  # e.g., SAM_vitH
                             current_split: str,                 # e.g., "train"
                             image_stems: list[str],             # list of image stems from batch["id"]
                             feature_keys: list[str],            # list of layer names to load, e.g., enc_layers
                             verbose: bool = True):
    """
    Loads precomputed teacher features, where each feature for each image is stored as a separate .npy file.
    Path structure: base_precomputed_dir/teacher_name/current_split/image_stem_sanitized_feature_key.npy
    Returns a list of tensors, one for each feature_key. Each tensor is (batch_size, *feature_dims).
    Raises FileNotFoundError or RuntimeError if features cannot be loaded or are inconsistent.
    """
    batched_features_for_each_key = []

    # Directory for the current teacher and split, e.g., ./precomputed/SAM_vitH/train/
    # This matches the output_base_dir / split_name structure from extract_teacher_features.py
    split_feature_dir = base_precomputed_dir / teacher_name / current_split

    for feature_key in feature_keys:
        # Sanitize the feature_key to match the filename format from extract_teacher_features.py
        sanitized_feature_key = feature_key.replace(".", "_").replace("[","_").replace("]","")
        
        tensors_for_current_key_in_batch = []
        path_of_last_attempted_file = None # For error reporting

        for img_stem in image_stems:
            feature_filename = f"{img_stem}_{sanitized_feature_key}.npy" #
            feature_path = split_feature_dir / feature_filename
            path_of_last_attempted_file = feature_path
            
            try:
                npy_array = np.load(feature_path)
                # The .squeeze(0) was done during saving if it was a batch of 1.
                # Here we load the per-image feature.
                tensors_for_current_key_in_batch.append(torch.from_numpy(npy_array).cuda(non_blocking=True))
            except FileNotFoundError:
                if verbose:
                    print(f"ERROR: Precomputed feature file NOT FOUND: {feature_path}")
                # Critical error: if one feature file in a batch is missing, we cannot form a complete batch for this key.
                raise FileNotFoundError(f"Required feature file missing: {feature_path}. Please ensure extract_teacher_features.py ran successfully for teacher '{teacher_name}', split '{current_split}'.")
            except Exception as e:
                if verbose:
                    print(f"ERROR: Could not load feature file {feature_path}: {e}")
                raise RuntimeError(f"Failed to load feature file {feature_path}") from e
        
        if tensors_for_current_key_in_batch:
            try:
                # Stack along batch dimension. All features for this key should have same dimensions.
                batched_features_for_each_key.append(torch.stack(tensors_for_current_key_in_batch))
            except Exception as e:
                 if verbose:
                    print(f"ERROR: Could not stack {len(tensors_for_current_key_in_batch)} features for key '{feature_key}'. Error: {e}. Attempted path pattern: {path_of_last_attempted_file}")
                 raise RuntimeError(f"Failed to stack features for key '{feature_key}'. Check for consistent shapes of precomputed features.") from e
        elif image_stems: # If image_stems is not empty but we have no tensors (should be caught by FileNotFoundError earlier)
             raise RuntimeError(f"No features loaded for key '{feature_key}' for the batch, though image_stems were provided. Last path: {path_of_last_attempted_file}")
        # If image_stems was empty, an empty list will be appended if feature_keys was non-empty,
        # or the loop over feature_keys won't run. The function will return an empty list if feature_keys is empty.

    if len(batched_features_for_each_key) != len(feature_keys):
        raise RuntimeError("Logic error: Mismatch between number of loaded feature groups and requested feature keys.")
        
    return batched_features_for_each_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- Dataset ----------

    ds_cfg = cfg["dataset"]
    train_root = ds_cfg["train_dataset"]   # "./datasets/train"
    val_root   = ds_cfg["val_dataset"]     # "./datasets/val"

    # 基本影像 / mask 轉換
    img_tfms  = T.ToTensor()
    mask_tfms = T.ToTensor()

    train_set = ComponentDataset(
        root_dir=train_root,
        transform=(img_tfms, mask_tfms),
        max_bbox_shift = ds_cfg.get("max_bbox_shift", 20), #
        prompt_mode    = ds_cfg.get("prompt_mode", "box"), #
        min_points     = ds_cfg.get("min_points", 1), #
        max_points     = ds_cfg.get("max_points", 3), #
        image_size     = cfg["model"].get("image_size", 1024) #
    )

    val_set   = ComponentDataset(
        root_dir=val_root,
        transform=(img_tfms, mask_tfms),
        max_bbox_shift = ds_cfg.get("max_bbox_shift", 20), #
        prompt_mode    = ds_cfg.get("prompt_mode", "box"), #
        min_points     = ds_cfg.get("min_points", 1), #
        max_points     = ds_cfg.get("max_points", 3), #
        image_size     = cfg["model"].get("image_size", 1024) #
    )

    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], #
                            shuffle=True, num_workers=ds_cfg.get("num_workers",4), #
                            pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=cfg["train"]["batch_size"], #
                            shuffle=False, num_workers=ds_cfg.get("num_workers",4), #
                            pin_memory=True)


    # ---------- Student ----------
    student_config_data = cfg.get("model") # 從載入的 JSON 設定檔中獲取 "model" 部分
    if student_config_data is None:
        raise ValueError("Model configuration ('model') not found in the config file.")

    student_model_type = student_config_data.get("type", "vit_t") #
    student_initial_checkpoint = student_config_data.get("checkpoint", None) #
                                                                          # 根據log, checkpoint是None, mobileSAM.json中是 "checkpoint_path"
                                                                          # 這邊的checkpoint指的是預訓練權重，例如SAM官方權重。
                                                                          # mobileSAM.json model.checkpoint_path 是 "./weights/mobile_sam.pt"
                                                                          # 若要載入 mobile_sam.pt, student_initial_checkpoint 應讀取此路徑

    # 修正：使用 "checkpoint_path" 而非 "checkpoint" 來載入學生模型的初始權重
    student_initial_checkpoint = student_config_data.get("checkpoint_path", None) #


    if student_model_type not in sam_model_registry:
        raise ValueError(f"Student model type '{student_model_type}' not found in sam_model_registry. "
                         f"Available types: {list(sam_model_registry.keys())}")
    
    model_builder_func = sam_model_registry[student_model_type]

    print(f"Building student model of type '{student_model_type}' with initial checkpoint: {student_initial_checkpoint}")
    student = model_builder_func(checkpoint=student_initial_checkpoint).to(device)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4) #

    # ---------- Hooks ----------
    # 注意：這些 layer name 可能不適用於 vit_t (MobileSAM)，需要依 MobileSAM 的結構調整以利蒸餾
    enc_layers = [f"image_encoder.blocks.{i}" for i in (9,10,11,12)] #
    dec_layer  = ["mask_decoder.pre_logits"] #
    attn_layers= [f"image_encoder.blocks.{i}.attn" for i in range(12)] #
    rkd_layer  = ["image_encoder.patch_embed"] #
    hook_names = enc_layers + dec_layer + attn_layers + rkd_layer #
    hook_handles = register_hooks(student, hook_names) #

    # ---------- Early Stopping ----------
    best_metric = -1
    patience = cfg["train"].get("early_stop_patience", 0) #
    stop_counter = 0

    for epoch in range(cfg["train"]["epochs"]): #
        student.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # --- MODIFICATION START: Input preparation for SAM model ---
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)
            ids = batch["id"]
            original_sizes = batch["original_size"] # List of tuples (H, W)

            batched_input_list = []
            for i in range(images.shape[0]):
                input_dict = {
                    "image": images[i],
                    "original_size": original_sizes[i] # SAM.forward expects original H, W
                }
                if "box" in batch and batch["box"][i] is not None:
                    input_dict["boxes"] = batch["box"][i].to(device) # Shape (1, 4) or (N, 4)
                if "points" in batch and batch["points"][i] is not None:
                    input_dict["point_coords"] = batch["points"][i].to(device) # Shape (N, 2)
                    if "labels" in batch and batch["labels"][i] is not None:
                        input_dict["point_labels"] = batch["labels"][i].to(device) # Shape (N,)
                batched_input_list.append(input_dict)
            
            optimizer.zero_grad()
            
            # Call student model with prepared batched_input and multimask_output flag
            # Sam.forward returns a list of dictionaries, one for each input image
            model_outputs = student(batched_input=batched_input_list, multimask_output=False)

            # Extract low_res_logits and upsample them for loss calculation
            # Assuming multimask_output=False, low_res_logits should be (B, 1, H_low, W_low)
            pred_logits_list = [out["low_res_logits"] for out in model_outputs]
            pred_logits = torch.stack(pred_logits_list) # Shape: (B, 1, H_low, W_low)
            
            # Upsample predicted logits to match ground truth mask resolution
            pred_logits_upsampled = F.interpolate(
                pred_logits,
                size=gt_masks.shape[-2:], # Target H, W from ground truth masks
                mode="bilinear",
                align_corners=False
            )
            # --- MODIFICATION END ---

            # --- task loss (Dice + BCE) ---
            # Use upsampled logits for loss calculation
            bce = F.binary_cross_entropy_with_logits(pred_logits_upsampled, gt_masks)
            inter = (torch.sigmoid(pred_logits_upsampled) * gt_masks).sum(dim=(-2, -1)) * 2 # sum over H, W
            union = torch.sigmoid(pred_logits_upsampled).sum(dim=(-2, -1)) + gt_masks.sum(dim=(-2, -1)) + 1e-6 # sum over H, W
            dice = 1 - (inter / union).mean() # Mean over batch and channels (if any)
            task_loss = 20 * bce + dice # Original weights from your script

            # --- distillation ---
            dist_loss = torch.tensor(0.0, device=device)
            if cfg["distillation"]["enable"]: #
                # Ensure features are popped only if hooks were successfully registered and distillation enabled
                # This part might still have issues if hook names are incorrect for vit_t (student)
                student_features_available = True
                try:
                    feats_s = pop_features() # Student features
                    if not feats_s: # If pop_features returns empty
                        # print("Warning: Student features (feats_s) are empty. Hooks might not be registered or working correctly.")
                        student_features_available = False
                except Exception as e:
                    # print(f"Warning: Could not pop student features: {e}. Skipping distillation.")
                    student_features_available = False

                if student_features_available:
                    # base_precomputed_dir typically from cfg, e.g. "./precomputed"
                    base_precomp_dir = pathlib.Path(cfg["distillation"]["precomputed_root"]) #
                    teacher_info = cfg["teachers"][0] # Assuming first teacher
                    te_name = teacher_info["name"] #
                    
                    teacher_model_type = ""
                    try:
                        with open(teacher_info["cfg"], 'r') as f_yaml: #
                            teacher_yaml_config = yaml.safe_load(f_yaml)
                        teacher_model_type = teacher_yaml_config.get("model", {}).get("type", "")
                    except Exception as e:
                        print(f"Warning: Could not load or parse teacher YAML config {teacher_info['cfg']}: {e}")

                    current_epoch_dist_loss = torch.tensor(0.0, device=device)

                    # ---- Encoder Matching ----
                    if cfg["distillation"]["encoder_matching"]["enable"]: #
                        # enc_layers from train.py: [f"image_encoder.blocks.{i}" for i in (9,10,11,12)]
                        # Check if all required student encoder layers are present in feats_s
                        if all(layer in feats_s for layer in enc_layers):
                            teacher_enc_keys_to_load = enc_layers
                            is_combined_vit_hl_encoder_load = False

                            # Special handling for ViT-H/L combined encoder features
                            # Based on extract_teacher_features.py, this applies if blocks 9,10,11,12 are the ones targeted.
                            if teacher_model_type in ['vit_h', 'vit_l'] and \
                               set(enc_layers) == {f"image_encoder.blocks.{i}" for i in (9,10,11,12)}:
                                teacher_enc_keys_to_load = ["image_encoder_blocks_9_12_combined"] #
                                is_combined_vit_hl_encoder_load = True
                            
                            try:
                                # image_stems for the current batch come from 'ids'
                                loaded_te_enc_features_list = load_cached_npy_features(
                                    base_precomp_dir, te_name, "train", ids, teacher_enc_keys_to_load
                                )
                                
                                final_teacher_enc_for_loss = []
                                if is_combined_vit_hl_encoder_load:
                                    if loaded_te_enc_features_list and loaded_te_enc_features_list[0].nelement() > 0:
                                        combined_tensor = loaded_te_enc_features_list[0] # Shape (num_combined_layers, B, D, H, W) after stack, or (B, num_combined_layers, D, H, W) if saved differently
                                                                                         # extract_teacher_features.py saves (4, D, H, W) per image.
                                                                                         # load_cached_npy_features stacks them to (B, 4, D, H, W) for the batch.
                                        if combined_tensor.shape[1] == len(enc_layers): # Check if 2nd dim is num_layers (4)
                                            # Unstack along the dimension that represents the different layers (dim 1)
                                            final_teacher_enc_for_loss = list(torch.unbind(combined_tensor, dim=1))
                                        else:
                                            print(f"Warning: Combined teacher enc feature for {te_name} has unexpected shape {combined_tensor.shape}, expected 2nd dim {len(enc_layers)}. Skipping enc matching.")
                                    else:
                                         print(f"Warning: Combined teacher enc feature for {te_name} was empty or not loaded. Skipping enc matching.")
                                else: # Not combined, or not ViT-H/L matching the specific combined case
                                    final_teacher_enc_for_loss = loaded_te_enc_features_list

                                if final_teacher_enc_for_loss and len(final_teacher_enc_for_loss) == len(enc_layers):
                                    current_epoch_dist_loss += encoder_matching_loss(
                                        [feats_s[l] for l in enc_layers], final_teacher_enc_for_loss, #
                                        **cfg["distillation"]["encoder_matching"] #
                                    )
                                elif final_teacher_enc_for_loss: # Mismatch in number
                                     print(f"Warning: Mismatch in num loaded teacher enc features ({len(final_teacher_enc_for_loss)}) vs expected ({len(enc_layers)}). Skipping.")
                                     
                            except (FileNotFoundError, RuntimeError) as e:
                                print(f"Warning: Skipping encoder_matching for {te_name} due to: {e}")
                        else:
                            print(f"Warning: Not all student encoder features found in feats_s ({[l for l in enc_layers if l not in feats_s]} missing). Skipping encoder_matching.")

                    # ---- Decoder Matching ----
                    if cfg["distillation"]["decoder_matching"]["enable"]: #
                        # dec_layer from train.py: ["mask_decoder.pre_logits"]
                        if dec_layer[0] in feats_s:
                            try:
                                # load_cached_npy_features returns a list, so take [0] for single key
                                te_dec_feature = load_cached_npy_features(
                                    base_precomp_dir, te_name, "train", ids, dec_layer
                                )[0]
                                if te_dec_feature.nelement() > 0:
                                    current_epoch_dist_loss += decoder_matching_loss(
                                        feats_s[dec_layer[0]], te_dec_feature, #
                                        **cfg["distillation"]["decoder_matching"] #
                                    )
                            except (FileNotFoundError, RuntimeError, IndexError) as e: # IndexError if list is empty
                                print(f"Warning: Skipping decoder_matching for {te_name} due to: {e}")
                        else:
                             print(f"Warning: Student decoder feature {dec_layer[0]} not in feats_s. Skipping decoder_matching.")

                    # ---- Attention Matching ----
                    # attn_layers from train.py: [f"image_encoder.blocks.{i}.attn" for i in range(12)]
                    # Note: extract_teacher_features.py does not seem to have special combined logic for these raw attention map names.
                    # It might capture them if they are part of "common_targets" or derived from block features if requested.
                    # Your current extract_teacher_features.py has common_targets = ["mask_decoder.transformer", "image_encoder.patch_embed"]
                    # And for ViT-H/L, it captures "image_encoder.blocks.{idx}". It does *not* explicitly capture ".attn" sub-features.
                    # This part of distillation might not work unless capture_targets in extract_teacher_features.py is updated
                    # to include these specific attention layers, or if they are implicitly saved.
                    # For now, assuming they might be saved under these exact keys by some mechanism.
                    if cfg["distillation"]["attention_matching"]["enable"]: #
                        if all(layer in feats_s for layer in attn_layers):
                            try:
                                te_attn_features = load_cached_npy_features(
                                    base_precomp_dir, te_name, "train", ids, attn_layers
                                )
                                # Ensure all features were loaded (list contains no None or empty tensors if loader was modified to return them)
                                if te_attn_features and all(t.nelement() > 0 for t in te_attn_features) and len(te_attn_features) == len(attn_layers):
                                    current_epoch_dist_loss += attention_matching_loss(
                                        [feats_s[l] for l in attn_layers], te_attn_features, #
                                        **cfg["distillation"]["attention_matching"] #
                                    )
                                elif te_attn_features:
                                     print(f"Warning: Not all teacher attention features loaded correctly ({len(te_attn_features)} vs {len(attn_layers)} expected). Skipping.")
                            except (FileNotFoundError, RuntimeError) as e:
                                print(f"Warning: Skipping attention_matching for {te_name} due to: {e}")
                        else:
                            print(f"Warning: Not all student attention features found in feats_s. Skipping attention_matching.")

                    # ---- Relational KD ----
                    # rkd_layer from train.py: ["image_encoder.patch_embed"]
                    if cfg["distillation"]["relational_KD"]["enable"]: #
                        if rkd_layer[0] in feats_s:
                            try:
                                te_rkd_feature = load_cached_npy_features(
                                    base_precomp_dir, te_name, "train", ids, rkd_layer
                                )[0]
                                if te_rkd_feature.nelement() > 0:
                                    current_epoch_dist_loss += rkd_loss(
                                        feats_s[rkd_layer[0]], te_rkd_feature, #
                                        **cfg["distillation"]["relational_KD"] #
                                    )
                            except (FileNotFoundError, RuntimeError, IndexError) as e:
                                print(f"Warning: Skipping relational_KD for {te_name} due to: {e}")
                        else:
                            print(f"Warning: Student RKD feature {rkd_layer[0]} not in feats_s. Skipping relational_KD.")
                    
                    dist_loss = current_epoch_dist_loss

            total_loss = task_loss + dist_loss
            total_loss.backward()
            optimizer.step()
            
            # Ensure dist_loss is a scalar for .item()
            dist_loss_item = dist_loss.item() if torch.is_tensor(dist_loss) else float(dist_loss)
            pbar.set_postfix({"task": task_loss.item(), "dist": dist_loss_item, "tot": total_loss.item()})


        # ---------- validation ----------
        student.eval()
        dices = []
        with torch.no_grad():
            for batch in val_loader: # val_loader also returns a batch dictionary
                # --- MODIFICATION START: Input preparation for SAM model in validation ---
                images = batch["image"].to(device)
                gt_masks = batch["mask"].to(device)
                original_sizes = batch["original_size"]

                batched_input_list_val = []
                for i in range(images.shape[0]):
                    input_dict_val = {
                        "image": images[i],
                        "original_size": original_sizes[i]
                    }
                    if "box" in batch and batch["box"][i] is not None:
                        input_dict_val["boxes"] = batch["box"][i].to(device)
                    if "points" in batch and batch["points"][i] is not None:
                        input_dict_val["point_coords"] = batch["points"][i].to(device)
                        if "labels" in batch and batch["labels"][i] is not None:
                            input_dict_val["point_labels"] = batch["labels"][i].to(device)
                    batched_input_list_val.append(input_dict_val)
                
                val_model_outputs = student(batched_input=batched_input_list_val, multimask_output=False)
                
                val_pred_logits_list = [out["low_res_logits"] for out in val_model_outputs]
                val_pred_logits = torch.stack(val_pred_logits_list)
                
                val_pred_logits_upsampled = F.interpolate(
                    val_pred_logits,
                    size=gt_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                # --- MODIFICATION END ---

                # Calculate Dice score using upsampled logits
                # Note: The original code used `pr` directly. Here we use `val_pred_logits_upsampled`.
                inter = (torch.sigmoid(val_pred_logits_upsampled) * gt_masks).sum(dim=(-2,-1)) * 2
                union = torch.sigmoid(val_pred_logits_upsampled).sum(dim=(-2,-1)) + gt_masks.sum(dim=(-2,-1)) + 1e-6
                dices.append((inter/union).mean().item())

        val_dice = float(np.mean(dices)) if dices else 0.0 # Handle case where dices might be empty
        print(f"Epoch {epoch}  ▸  val Dice={val_dice:.4f}")

        # ---------- early‑stop monitor ----------
        if patience > 0:
            if val_dice > best_metric:
                best_metric = val_dice
                stop_counter = 0
                # Consider saving based on cfg["model"]["save_path"] if defined
                save_dir = pathlib.Path(cfg["model"].get("save_path", ".")) #
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(student.state_dict(), save_dir / "best_student.pth")
                print(f"Saved new best model to {save_dir / 'best_student.pth'}")
            else:
                stop_counter += 1
                if stop_counter >= patience:
                    print(f"Early stopping triggered ▸ best Dice={best_metric:.4f}")
                    break

    for h in hook_handles:
        h.remove()

if __name__ == "__main__":
    main()