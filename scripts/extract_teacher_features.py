"""
假設您的根目錄是 MobileSAM-fast-finetuning，並且您的資料集結構如您所述 (datasets/train/image/..., datasets/val/image/...)。

您可以使用類似以下的指令：

Bash

python scripts/extract_teacher_features.py \
    --teacher_name SAM_vitH \
    --teacher_cfg ./configs/sam_vith.yaml \
    --teacher_ckpt ./weights/sam_vit_h_4b8939.pth \
    --dataset_base_dir ./datasets \
    --output_base_dir ./precomputed/SAM_vitH \
    --splits train val \
    --image_subdir_name image

python scripts/extract_teacher_features.py \
    --teacher_name MobileSAM_orig \
    --teacher_cfg configs/mobile_sam_orig.yaml \
    --teacher_ckpt weights/mobile_sam.pt \
    --dataset_base_dir ./datasets \
    --output_base_dir ./precomputed/MobileSAM_orig \
    --splits train val \
    --image_subdir_name image
這個指令會：

處理 ./datasets/train/image/ 下的所有圖片，並將特徵儲存在 ./precomputed/MobileSAM_orig/train/。
處理 ./datasets/val/image/ 下的所有圖片，並將特徵儲存在 ./precomputed/MobileSAM_orig/val/。
如果您只想處理 train split：

Bash

python scripts/extract_teacher_features.py \
    --teacher_name MobileSAM_orig \
    --teacher_cfg configs/mobile_sam_orig.yaml \
    --teacher_ckpt weights/mobile_sam.pt \
    --dataset_base_dir ./datasets \
    --output_base_dir ./precomputed/MobileSAM_orig \
    --splits train \
    --image_subdir_name image
"""

import argparse, pathlib, numpy as np, torch
from PIL import Image
import torchvision.transforms as T
from finetune_utils.feature_hooks import register_hooks, pop_features #
import yaml

from mobile_sam import sam_model_registry #

def process_images_in_dir(model, image_dir_path: pathlib.Path, output_base_dir: pathlib.Path, capture_targets: list, device: str, model_type: str): # 新增 model_type 參數
    """
    處理指定目錄中的所有圖片，並將提取的特徵保存到指定的輸出位置。
    對於 ViT-H/L，特定的 image_encoder.blocks 會被合併。
    """
    if not image_dir_path.is_dir():
        print(f"Info: Image directory {image_dir_path} does not exist or is not a directory. Skipping.")
        return

    current_output_dir = output_base_dir
    current_output_dir.mkdir(parents=True, exist_ok=True)
    
    img_extensions = ["*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]", "*.[pP][nN][gG]"]
    imgs = []
    for ext in img_extensions:
        imgs.extend(list(image_dir_path.glob(ext)))
    imgs = sorted(list(set(imgs)))

    if not imgs:
        print(f"Warning: No images (jpg, jpeg, png) found in {image_dir_path}.")
        return

    print(f"Processing {len(imgs)} images from {image_dir_path}...")
    
    handles = register_hooks(model, capture_targets)

    # 定義 ViT-H/L 中要合併的 image encoder block 特徵的鍵名
    # 這些鍵名應該與 capture_targets 中為 vit_h/vit_l 生成的鍵名一致
    vit_block_keys_to_combine = [
        "image_encoder.blocks.9",
        "image_encoder.blocks.10",
        "image_encoder.blocks.11",
        "image_encoder.blocks.12"
    ]

    for p in imgs:
        try:
            pil_img = Image.open(p).convert("RGB")
            original_h, original_w = pil_img.height, pil_img.width
            
            img_array = np.array(pil_img) 
            img_for_preprocess = torch.as_tensor(img_array, dtype=torch.float32).permute(2, 0, 1).to(device)

            batched_item = {
                "image": img_for_preprocess,
                "original_size": (original_h, original_w)
            }
            current_batched_input = [batched_item]
            multimask_output_setting = False 

            with torch.no_grad():
                _ = model.forward(current_batched_input, multimask_output_setting)
            
            feats = pop_features()
            
            temp_numpy_features = {} # 暫存所有轉換為 numpy 的特徵
            for key, captured_list_of_features in feats.items():
                if captured_list_of_features: 
                    feature_tensor = captured_list_of_features[0] 
                    if isinstance(feature_tensor, torch.Tensor):
                        temp_numpy_features[key] = feature_tensor.cpu().squeeze(0).numpy()
                    else:
                        print(f"Warning: Captured item for key '{key}' is not a tensor, it's a {type(feature_tensor)} for image {p.name}.")
                else:
                    print(f"Warning: No features captured in the list for key '{key}' for image {p.name}.")

            final_save_items = {} # 最終要儲存的項目 (鍵名 -> numpy 陣列)
            collected_block_features_for_stacking = [] # 收集用於堆疊的特徵

            # 檢查是否為 ViT-H/L 並嘗試合併指定的 block 特徵
            attempt_combination = model_type in ['vit_h', 'vit_l']
            
            if attempt_combination:
                can_combine_all_specific_blocks = True
                for bk_key in vit_block_keys_to_combine:
                    if bk_key in temp_numpy_features:
                        collected_block_features_for_stacking.append(temp_numpy_features[bk_key])
                    else:
                        # 即使 capture_targets 中包含了這些鍵，也可能因為某些原因 (例如模型實際層數不足) 導致未捕獲到
                        # 檢查 capture_targets 中是否真的包含了這些鍵 (對於 vit_b 可能不全包含)
                        if bk_key in capture_targets:
                            print(f"Warning: Expected feature '{bk_key}' was in capture_targets but not found in extracted feats for {p.name}. Cannot combine ViT blocks.")
                        can_combine_all_specific_blocks = False
                        break 
                
                if can_combine_all_specific_blocks and len(collected_block_features_for_stacking) == len(vit_block_keys_to_combine):
                    try:
                        combined_name = "image_encoder_neck"
                        # 沿著新的第0維堆疊: (4, original_dims...)
                        final_save_items[combined_name] = np.stack(collected_block_features_for_stacking, axis=0)
                        print(f"Info: Combined {len(vit_block_keys_to_combine)} ViT block features into '{combined_name}' for {p.name}.")
                        # 從 temp_numpy_features 中移除已合併的單獨 block 特徵，避免重複儲存
                        for bk_key in vit_block_keys_to_combine:
                            temp_numpy_features.pop(bk_key, None)
                    except Exception as e:
                        print(f"Error combining ViT block features for {p.name}: {e}. Will save them individually if available.")
                elif attempt_combination : # 即使是 vit_h/l，如果未能收集齊全部指定的 block，則不合併
                    print(f"Info: Not all specified ViT blocks ({len(vit_block_keys_to_combine)} expected, {len(collected_block_features_for_stacking)} found) were available for combination for {p.name}. Saving individually if present.")


            # 將剩餘的 (或未被成功合併的) 特徵加入到最終儲存列表
            for key, numpy_feature in temp_numpy_features.items():
                final_save_items[key] = numpy_feature
            
            # 儲存 final_save_items 中的所有特徵為獨立的 .npy 檔案
            num_features_saved_for_image = 0
            if not final_save_items:
                 print(f"No features processed to save for {p.name}.")
            else:
                for key, numpy_feature_to_save in final_save_items.items():
                    sanitized_key = key.replace(".", "_").replace("[","_").replace("]","") # 清理鍵名作為檔案名的一部分
                    feature_filename = f"{p.stem}_{sanitized_key}.npy"
                    feature_save_path = current_output_dir / feature_filename
                    try:
                        np.save(feature_save_path, numpy_feature_to_save)
                        num_features_saved_for_image += 1
                    except Exception as e:
                        print(f"Error saving feature '{key}' for {p.name} to {feature_save_path}: {e}")

            if num_features_saved_for_image > 0:
                print(f"Saved {num_features_saved_for_image} feature file(s) for {p.name} in {current_output_dir}")
            else:
                print(f"No features were ultimately saved for {p.name}.")

        except Exception as e:
            print(f"Critical error processing file {p.name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            pass 

    for h in handles: 
        h.remove()
    print(f"Finished processing images in {image_dir_path}. Features potentially saved to {current_output_dir}")


def main():
    ap = argparse.ArgumentParser(description="Extract features from a teacher model for specified dataset splits.")
    ap.add_argument("--teacher_name", required=True, help="Name of the teacher model.")
    ap.add_argument("--teacher_cfg", required=True, help="Path to the teacher model's YAML configuration file.")
    ap.add_argument("--teacher_ckpt", required=True, help="Path to the teacher model's checkpoint file.")
    ap.add_argument("--dataset_base_dir", required=True, help="Base directory for the datasets (e.g., './datasets').")
    ap.add_argument("--output_base_dir", required=True, help="Base directory for saving precomputed features (e.g., './precomputed/MobileSAM_orig'). The script will create subdirectories for each split here.")
    ap.add_argument("--splits", nargs='+', default=['train', 'val'], help="List of dataset splits to process (e.g., 'train' 'val'). Default: ['train', 'val']")
    ap.add_argument("--image_subdir_name", default="image", help="Subdirectory name under each split containing images (e.g., 'image'). Default: 'image'")
    ap.add_argument("--device", default="cuda", help="Device to use for computation (e.g., 'cuda', 'cpu'). Default: 'cuda'")
    args = ap.parse_args()

    with open(args.teacher_cfg, 'r') as f:
        teacher_config = yaml.safe_load(f)
    
    model_config_dict = teacher_config.get('model')
    if model_config_dict is None:
        raise ValueError(f"Error: 'model' key not found in the teacher configuration file: {args.teacher_cfg}")
    
    model_type = model_config_dict.get('type')
    if model_type is None:
        raise ValueError(f"Error: 'type' (for model_type) not found under 'model' key in the teacher configuration file: {args.teacher_cfg}")

    # --- 動態定義 capture_targets 列表 ---
    capture_targets = []
    common_targets = ["mask_decoder.transformer", "image_encoder.patch_embed"] #

    if model_type == 'vit_t': # MobileSAM (TinyViT)
        encoder_feature_targets = ["image_encoder.neck"] #
        capture_targets.extend(encoder_feature_targets)
        capture_targets.extend(common_targets)
    elif model_type in ['vit_b', 'vit_l', 'vit_h']: # 標準 SAM ViT 模型
        encoder_block_targets = []
        requested_indices = [9, 10, 11, 12] 
        max_idx = -1
        if model_type == 'vit_b': max_idx = 11
        elif model_type == 'vit_l': max_idx = 23
        elif model_type == 'vit_h': max_idx = 31
        valid_requested_indices = [idx for idx in requested_indices if idx <= max_idx]
        if not valid_requested_indices and max_idx >=0 :
            num_blocks_to_capture = min(3, max_idx + 1)
            valid_requested_indices = [(max_idx - i) for i in range(num_blocks_to_capture)]
            valid_requested_indices.reverse()
        for idx in sorted(list(set(valid_requested_indices))):
            encoder_block_targets.append(f"image_encoder.blocks.{idx}")
        capture_targets.extend(encoder_block_targets)
        capture_targets.extend(common_targets)
    else:
        raise ValueError(f"Model type '{model_type}' is not recognized for defining feature capture list.")
    # --- 動態定義結束 ---

    try:
        model_builder = sam_model_registry[model_type]
    except KeyError:
        raise ValueError(f"Error: Unsupported 'model_type' ('{model_type}') in {args.teacher_cfg}. Available: {list(sam_model_registry.keys())}")

    print(f"Building model of type '{model_type}' with checkpoint '{args.teacher_ckpt}' for teacher '{args.teacher_name}'")
    model = model_builder(checkpoint=args.teacher_ckpt).to(args.device).eval()
    
    # Hooks 只需要在模型建立後註冊一次，除非模型或 capture_targets 改變
    # 但由於 _FEATURE_STORE 是全域的，且在 register_hooks 中清空，
    # 如果 process_images_in_dir 被多次調用（例如對 train, val），
    # 則 hooks 應在 process_images_in_dir 內部，或者 register_hooks 不清空STORE，由 pop 清空。
    # 目前的 feature_hooks.py: register_hooks 清空 STORE, pop_features 也清空 STORE。
    # 這意味著對於每個 split，我們需要重新註冊 hooks。
    # 或者，更好的做法是，讓 process_images_in_dir 接收 model，並在內部處理 hooks 的註冊和移除。
    # 我們之前的版本將 handles = register_hooks(model, capture_targets) 放在 process_images_in_dir 內部循環之外，這是正確的。
    # 但 handles 的移除也應與其對應。

    base_dataset_path = pathlib.Path(args.dataset_base_dir)
    output_root_path = pathlib.Path(args.output_base_dir)

    # 將 handles 的管理移到 process_images_in_dir 內部，或者確保其生命週期正確
    # 為了簡潔，我們假設 process_images_in_dir 內的 hooks 管理是正確的（如上一版本所示，在循環外註冊，循環後移除）
    # 此處修改為：在 process_images_in_dir 的開頭註冊，結尾移除。

    for split_name in args.splits:
        print(f"\nProcessing split: {split_name}")
        image_dir_path = base_dataset_path / split_name / args.image_subdir_name
        split_output_dir = output_root_path / split_name 
        
        print(f"Preparing to process images for split '{split_name}'. Using features: {capture_targets}")
        # 將 model_type (或您在 main 中使用的變數名) 傳遞給 process_images_in_dir
        process_images_in_dir(model, image_dir_path, split_output_dir, capture_targets, args.device, model_type) # <--- 確認這裡傳遞了 model_type

    print("\nFeature extraction process completed.")

if __name__ == "__main__":
    main()