# MobileSAM Segment-Everything Fine-tune — Hybrid Teacher / Ground-Truth

本文件說明如何以「格點提示 (grid prompts)」搭配 *Teacher → Ground-Truth 替換* 的策略，微調 (fine-tune) MobileSAM 使其具備 **Segment Everything** 能力。

## 核心概念
1. **格點提示** ‑ 於影像上均勻灑下 \(N\times N\) 個點作為提示，每個點由解碼器輸出三個候選遮罩 (multimask)。
2. **Teacher 模型** ‑ 使用原始 MobileSAM 或 SAM-H 做為教師，提供遮罩作為弱標註 (pseudo-label)。
3. **Ground-Truth 覆寫** ‑ 若 *教師遮罩* 與任何 *GT 物件* 的 IoU ≥ `mask_teacher.iou_threshold` (預設 0.5)，且格點位於該 GT 物件內，則以 **GT 遮罩** 取代教師遮罩；其餘遮罩仍採教師結果。
4. **逐遮罩配對** ‑ 學生與參考遮罩（Teacher / GT）以 Hungarian matching 配對，只計算配對成功且 IoU ≥ 0.3 的樣本以避免梯度稀釋。
5. **損失函式**
   * Binary Cross-Entropy、Focal、Dice
   * IoU Regression
   * 遮罩品質分類 (confidence)
   * 可選：SPA-KD distillation / Dense Mask distillation
6. **Multi-mask** — 訓練時開啟 `multimask_output=True` 以保留三種候選遮罩，可在推論階段僅取 IoU 最高者以減少運算。

## 設定檔 (configs/mobileSAM_se.json)
```
"mask_teacher": {
  "enable": true,
  "method": "replace_gt",
  "teacher_type": "vit_h",
  "checkpoint_path": "weights/sam_vit_h_4b8939.pth",
  "iou_threshold": 0.5,
  "gt_override": true
}
```
> *`method`* 設為 `replace_gt`，程式將自動使用「Teacher → GT 覆寫」邏輯。

## 執行步驟
1. 準備資料集結構
   ```
   dataset/
     image/XXX.jpg
     mask/XXX/object_0.png
     mask/XXX/object_1.png
   ```
2. 調整 `configs/mobileSAM_se.json` 中路徑與訓練超參數。
3. 啟動訓練
   ```bash
   python train.py --config configs/mobileSAM_se.json
   ```
4. 最佳模型將存於 `logs/best_student.pth`，TensorBoard 日誌亦位於相同目錄。

## 注意事項
* GPU 記憶體需求與 `grid_points` 二次方成長；若 OOM 請降低點數或影像解析度。
* `batch_size=1` 已足以使用 24 GB VRAM，在混合精度 (bfloat16) 下可再降低約 20% 記憶體。
* 選用 `dual_supervision` 時，程式仍會額外套用 soft loss (MSE / KL) 於教師與學生 logits。

---
最後更新：2025-06-29 