import yaml 
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


def load_cached_npy_features(base_precomputed_dir: pathlib.Path,
                             teacher_name: str,
                             current_split: str,
                             image_stems: list[str],
                             feature_keys: list[str],
                             verbose: bool = True):
    batched_features_for_each_key = []
    split_feature_dir = base_precomputed_dir / teacher_name / current_split

    for feature_key in feature_keys:
        sanitized_feature_key = feature_key.replace(".", "_").replace("[","_").replace("]","")
        tensors_for_current_key_in_batch = []
        path_of_last_attempted_file = None
        for img_stem in image_stems:
            feature_filename = f"{img_stem}_{sanitized_feature_key}.npy"
            feature_path = split_feature_dir / feature_filename
            path_of_last_attempted_file = feature_path
            try:
                npy_array = np.load(feature_path)
                tensors_for_current_key_in_batch.append(torch.from_numpy(npy_array).cuda(non_blocking=True))
            except FileNotFoundError:
                if verbose:
                    print(f"ERROR: Precomputed feature file NOT FOUND: {feature_path}")
                raise FileNotFoundError(f"Required feature file missing: {feature_path}. Please ensure extract_teacher_features.py ran successfully for teacher '{teacher_name}', split '{current_split}'.")
            except Exception as e:
                if verbose:
                    print(f"ERROR: Could not load feature file {feature_path}: {e}")
                raise RuntimeError(f"Failed to load feature file {feature_path}") from e
        
        if tensors_for_current_key_in_batch:
            try:
                batched_features_for_each_key.append(torch.stack(tensors_for_current_key_in_batch))
            except Exception as e:
                if verbose:
                    print(f"ERROR: Could not stack {len(tensors_for_current_key_in_batch)} features for key '{feature_key}'. Error: {e}. Attempted path pattern: {path_of_last_attempted_file}")
                raise RuntimeError(f"Failed to stack features for key '{feature_key}'. Check for consistent shapes of precomputed features.") from e
        elif image_stems: 
             raise RuntimeError(f"No features loaded for key '{feature_key}' for the batch, though image_stems were provided. Last path: {path_of_last_attempted_file}")

    if len(batched_features_for_each_key) != len(feature_keys):
        if not image_stems and feature_keys:
             pass 
        elif image_stems: 
            raise RuntimeError(f"Logic error: Mismatch between number of loaded feature groups ({len(batched_features_for_each_key)}) and requested feature keys ({len(feature_keys)}).")
            
    return batched_features_for_each_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_cfg = cfg["dataset"]
    train_root = ds_cfg["train_dataset"]
    val_root   = ds_cfg["val_dataset"]
    img_tfms  = T.ToTensor()
    mask_tfms = T.ToTensor()

    train_set = ComponentDataset(
        root_dir=train_root, transform=(img_tfms, mask_tfms),
        max_bbox_shift=ds_cfg.get("max_bbox_shift", 20),
        prompt_mode=ds_cfg.get("prompt_mode", "box"),
        min_points=ds_cfg.get("min_points", 1),
        max_points=ds_cfg.get("max_points", 3),
        image_size=cfg["model"].get("image_size", 1024)
    )
    val_set = ComponentDataset(
        root_dir=val_root, transform=(img_tfms, mask_tfms),
        max_bbox_shift=ds_cfg.get("max_bbox_shift", 20),
        prompt_mode=ds_cfg.get("prompt_mode", "box"),
        min_points=ds_cfg.get("min_points", 1),
        max_points=ds_cfg.get("max_points", 3),
        image_size=cfg["model"].get("image_size", 1024)
    )
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"],
                            shuffle=True, num_workers=ds_cfg.get("num_workers",4),
                            pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg["train"]["batch_size"],
                            shuffle=False, num_workers=ds_cfg.get("num_workers",4),
                            pin_memory=True)

    student_config_data = cfg.get("model")
    if student_config_data is None:
        raise ValueError("Model configuration ('model') not found in the config file.")
    student_model_type = student_config_data.get("type", "vit_t")
    student_initial_checkpoint = student_config_data.get("checkpoint_path", None)

    if student_model_type not in sam_model_registry:
        raise ValueError(f"Student model type '{student_model_type}' not found. Available: {list(sam_model_registry.keys())}")
    model_builder_func = sam_model_registry[student_model_type]
    print(f"Building student model of type '{student_model_type}' with initial checkpoint: {student_initial_checkpoint}")
    student = model_builder_func(checkpoint=student_initial_checkpoint).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)

    # ---------- Hooks (REFINED SECTION) ----------
    # These are the *potential* layers we might want to hook, based on the student model type
    # and what extract_teacher_features.py typically captures for those types.
    potential_hooks = {
        "enc": [],
        "dec": [], # For vit_t, 'mask_decoder.pre_logits' doesn't exist. 'mask_decoder.transformer' is an option if loss handles it.
        "attn": [],
        "rkd": ["image_encoder.patch_embed"] # Generally available
    }

    if student_model_type == 'vit_t':
        print("Student model is vit_t (MobileSAM/TinyViT), selecting vit_t specific hook names.")
        potential_hooks["enc"] = ["image_encoder.neck"]
        # For vit_t, 'mask_decoder.pre_logits' is not applicable.
        # If decoder_matching is desired for vit_t, potential_hooks["dec"] could be set to ["mask_decoder.transformer"],
        # but the loss function (decoder_matching_loss) must be able to handle the output of a Transformer module.
        # To avoid the 'mask_decoder.pre_logits' warning, we leave it empty or use a known valid layer for vit_t's decoder.
        potential_hooks["dec"] = [] # Or specify a valid vit_t decoder layer if known and loss supports it.
        potential_hooks["attn"] = [] # Needs specific, verified paths for TinyViT attention layers.
    elif student_model_type in ['vit_b', 'vit_l', 'vit_h']:
        print(f"Student model is {student_model_type}, selecting standard ViT hook names (verify indices).")
        potential_hooks["enc"] = [f"image_encoder.blocks.{i}" for i in (9,10,11,12)]
        potential_hooks["dec"] = ["mask_decoder.pre_logits"] # Assumed for standard SAM
        potential_hooks["attn"] = [f"image_encoder.blocks.{i}.attn" for i in range(12)]
    else:
        print(f"Warning: Unknown student_model_type '{student_model_type}'. Using default (likely ViT-H like) hook names which may cause warnings.")
        potential_hooks["enc"] = [f"image_encoder.blocks.{i}" for i in (9,10,11,12)]
        potential_hooks["dec"] = ["mask_decoder.pre_logits"]
        potential_hooks["attn"] = [f"image_encoder.blocks.{i}.attn" for i in range(12)]

    # Construct the final list of hook names to register based on enabled distillation types
    hook_names_to_register = []
    distill_cfg = cfg.get("distillation", {})
    if distill_cfg.get("encoder_matching", {}).get("enable"):
        hook_names_to_register.extend(potential_hooks["enc"])
    if distill_cfg.get("decoder_matching", {}).get("enable"):
        hook_names_to_register.extend(potential_hooks["dec"])
    if distill_cfg.get("attention_matching", {}).get("enable"):
        hook_names_to_register.extend(potential_hooks["attn"])
    if distill_cfg.get("relational_KD", {}).get("enable"):
        hook_names_to_register.extend(potential_hooks["rkd"])
    
    hook_names_to_register = sorted(list(set(h for h in hook_names_to_register if h))) 

    hook_handles = []
    if distill_cfg.get("enable") and hook_names_to_register:
        print(f"Registering hooks for layers: {hook_names_to_register}")
        hook_handles = register_hooks(student, hook_names_to_register)
    elif distill_cfg.get("enable"):
        print("Warning: Distillation is enabled, but no hook names were determined for active distillation types. Hooks not registered.")
    # ---------- End of Hooks REFINED SECTION ----------

    best_metric = -1
    patience = cfg["train"].get("early_stop_patience", 0)
    stop_counter = 0

    for epoch in range(cfg["train"]["epochs"]):
        student.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)
            ids = batch["id"] # image stems
            original_sizes_batch_data = batch["original_size"] 

            batched_input_list = []
            for i in range(images.shape[0]):
                h_orig, w_orig = original_sizes_batch_data[i]
                current_original_size = (int(h_orig), int(w_orig)) 

                input_dict = {"image": images[i], "original_size": current_original_size}
                if "box_prompt" in batch and batch["box_prompt"][i] is not None:
                    input_dict["boxes"] = batch["box_prompt"][i].to(device)
                if "point_coords" in batch and batch["point_coords"][i] is not None:
                    input_dict["point_coords"] = batch["point_coords"][i].to(device)
                    if "point_labels" in batch and batch["point_labels"][i] is not None:
                        input_dict["point_labels"] = batch["point_labels"][i].to(device)
                batched_input_list.append(input_dict)
            
            optimizer.zero_grad()
            model_outputs = student(batched_input=batched_input_list, multimask_output=False)
            
            pred_logits_list = [out["low_res_logits"] for out in model_outputs]
            pred_logits = torch.stack(pred_logits_list, dim=0) 

            if pred_logits.ndim == 5 and pred_logits.shape[1] == 1: 
                pred_logits = pred_logits.squeeze(1)
            
            pred_logits_upsampled = F.interpolate(
                pred_logits, size=gt_masks.shape[-2:], mode="bilinear", align_corners=False
            )

            bce = F.binary_cross_entropy_with_logits(pred_logits_upsampled, gt_masks)
            inter = (torch.sigmoid(pred_logits_upsampled) * gt_masks).sum(dim=(-2, -1)) * 2
            union = torch.sigmoid(pred_logits_upsampled).sum(dim=(-2, -1)) + gt_masks.sum(dim=(-2, -1)) + 1e-6
            dice = 1 - (inter / union).mean()
            task_loss = 20 * bce + dice

            dist_loss = torch.tensor(0.0, device=device)
            if distill_cfg.get("enable") and hook_handles:
                student_features_available = True
                try:
                    feats_s = pop_features()
                    if not feats_s: student_features_available = False
                except Exception: student_features_available = False

                if student_features_available:
                    base_precomp_dir = pathlib.Path(distill_cfg["precomputed_root"])
                    teacher_info = cfg["teachers"][0]
                    te_name = teacher_info["name"]
                    teacher_model_type_cfg = ""
                    try:
                        with open(teacher_info["cfg"], 'r') as f_yaml:
                            teacher_yaml_config = yaml.safe_load(f_yaml)
                        teacher_model_type_cfg = teacher_yaml_config.get("model", {}).get("type", "")
                    except Exception as e_yaml: print(f"Warning: YAML load failed {teacher_info['cfg']}: {e_yaml}")

                    current_epoch_dist_loss = torch.tensor(0.0, device=device)
                    
                    # Encoder Matching
                    if distill_cfg.get("encoder_matching", {}).get("enable") and potential_hooks["enc"]:
                        # Use potential_hooks["enc"] as the list of student layers we expect features for
                        if all(layer in feats_s for layer in potential_hooks["enc"]):
                            teacher_keys_to_load = potential_hooks["enc"]
                            is_combined = False
                            if teacher_model_type_cfg in ['vit_h', 'vit_l'] and \
                               set(potential_hooks["enc"]) == {f"image_encoder.blocks.{i}" for i in (9,10,11,12)}:
                                teacher_keys_to_load = ["image_encoder_blocks_9_12_combined"]
                                is_combined = True
                            try:
                                loaded_te_feats = load_cached_npy_features(base_precomp_dir, te_name, "train", ids, teacher_keys_to_load)
                                final_te_for_loss = []
                                if is_combined:
                                    if loaded_te_feats and loaded_te_feats[0].nelement() > 0:
                                        combined_tensor = loaded_te_feats[0] 
                                        if combined_tensor.shape[1] == len(potential_hooks["enc"]): 
                                            final_te_for_loss = list(torch.unbind(combined_tensor, dim=1))
                                else: 
                                    final_te_for_loss = loaded_te_feats
                                
                                if final_te_for_loss and len(final_te_for_loss) == len(potential_hooks["enc"]):
                                    current_epoch_dist_loss += encoder_matching_loss(
                                        [feats_s[l] for l in potential_hooks["enc"]], final_te_for_loss,
                                        **distill_cfg["encoder_matching"])
                            except (FileNotFoundError, RuntimeError) as e_dist: print(f"Enc distill skip: {e_dist}")
                    
                    # Decoder Matching
                    if distill_cfg.get("decoder_matching", {}).get("enable") and potential_hooks["dec"]:
                        if potential_hooks["dec"] and potential_hooks["dec"][0] in feats_s : # Check if list not empty before indexing
                            try:
                                te_dec = load_cached_npy_features(base_precomp_dir, te_name, "train", ids, potential_hooks["dec"])[0]
                                if te_dec.nelement() > 0:
                                    current_epoch_dist_loss += decoder_matching_loss(
                                        feats_s[potential_hooks["dec"][0]], te_dec, **distill_cfg["decoder_matching"])
                            except (FileNotFoundError, RuntimeError, IndexError) as e_dist: print(f"Dec distill skip: {e_dist}")

                    # Attention Matching
                    if distill_cfg.get("attention_matching", {}).get("enable") and potential_hooks["attn"]:
                        if all(layer in feats_s for layer in potential_hooks["attn"]):
                            try:
                                te_attn = load_cached_npy_features(base_precomp_dir, te_name, "train", ids, potential_hooks["attn"])
                                if te_attn and all(t.nelement() > 0 for t in te_attn) and len(te_attn) == len(potential_hooks["attn"]):
                                     current_epoch_dist_loss += attention_matching_loss(
                                        [feats_s[l] for l in potential_hooks["attn"]], te_attn, **distill_cfg["attention_matching"])
                            except (FileNotFoundError, RuntimeError) as e_dist: print(f"Attn distill skip: {e_dist}")

                    # Relational KD
                    if distill_cfg.get("relational_KD", {}).get("enable") and potential_hooks["rkd"]: # potential_hooks["rkd"] is ["image_encoder.patch_embed"]
                        if potential_hooks["rkd"][0] in feats_s:
                            try:
                                te_rkd = load_cached_npy_features(base_precomp_dir, te_name, "train", ids, potential_hooks["rkd"])[0]
                                if te_rkd.nelement() > 0:
                                    current_epoch_dist_loss += rkd_loss(
                                        feats_s[potential_hooks["rkd"][0]], te_rkd, **distill_cfg["relational_KD"])
                            except (FileNotFoundError, RuntimeError, IndexError) as e_dist: print(f"RKD distill skip: {e_dist}")
                    dist_loss = current_epoch_dist_loss
            
            total_loss = task_loss + dist_loss
            total_loss.backward()
            optimizer.step()
            dist_loss_item = dist_loss.item() if torch.is_tensor(dist_loss) else float(dist_loss)
            pbar.set_postfix({"task": task_loss.item(), "dist": dist_loss_item, "tot": total_loss.item()})

        # Validation loop
        student.eval(); dices = []
        with torch.no_grad():
            for batch_val in val_loader: 
                images_val = batch_val["image"].to(device)
                gt_masks_val = batch_val["mask"].to(device)
                original_sizes_val_batch = batch_val["original_size"]

                batched_input_list_val = []
                for i in range(images_val.shape[0]):
                    h_orig_val, w_orig_val = original_sizes_val_batch[i]
                    current_original_size_val = (int(h_orig_val), int(w_orig_val))
                    input_dict_val = {"image": images_val[i], "original_size": current_original_size_val}
                    if "box_prompt" in batch_val and batch_val["box_prompt"][i] is not None:
                        input_dict_val["boxes"] = batch_val["box_prompt"][i].to(device)
                    if "point_coords" in batch_val and batch_val["point_coords"][i] is not None:
                        input_dict_val["point_coords"] = batch_val["point_coords"][i].to(device)
                        if "point_labels" in batch_val and batch_val["point_labels"][i] is not None:
                            input_dict_val["point_labels"] = batch_val["point_labels"][i].to(device)
                    batched_input_list_val.append(input_dict_val)
                
                val_model_outputs = student(batched_input=batched_input_list_val, multimask_output=False)
                val_pred_logits_list = [out["low_res_logits"] for out in val_model_outputs]
                val_pred_logits = torch.stack(val_pred_logits_list, dim=0)

                if val_pred_logits.ndim == 5 and val_pred_logits.shape[1] == 1:
                    val_pred_logits = val_pred_logits.squeeze(1)
                
                val_pred_logits_upsampled = F.interpolate(
                    val_pred_logits, size=gt_masks_val.shape[-2:], mode="bilinear", align_corners=False)
                
                inter_val = (torch.sigmoid(val_pred_logits_upsampled) * gt_masks_val).sum(dim=(-2,-1)) * 2
                union_val = torch.sigmoid(val_pred_logits_upsampled).sum(dim=(-2,-1)) + gt_masks_val.sum(dim=(-2,-1)) + 1e-6
                dices.append((inter_val/union_val).mean().item())
        val_dice = float(np.mean(dices)) if dices else 0.0
        print(f"Epoch {epoch}  ▸  val Dice={val_dice:.4f}")

        if patience > 0:
            if val_dice > best_metric:
                best_metric = val_dice; stop_counter = 0
                save_dir = pathlib.Path(cfg["model"].get("save_path", "."))
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(student.state_dict(), save_dir / "best_student.pth")
                print(f"Saved new best model to {save_dir / 'best_student.pth'}")
            else:
                stop_counter += 1
                if stop_counter >= patience:
                    print(f"Early stopping triggered ▸ best Dice={best_metric:.4f}"); break
    
    if hook_handles: 
        for h in hook_handles: h.remove()

if __name__ == "__main__":
    main()