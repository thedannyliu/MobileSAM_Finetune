# test_smoke.py
import torch
from torch.utils.data import DataLoader
from mobilesam import build_model
from finetune_utils.dataset import ComponentDataset
from train import main  # 或者直接調用 train.py 的函式

def smoke_run():
    # 1. 準備極小資料集（假設 datasets/toy/train 下只放 2 張圖+mask）
    ds = ComponentDataset(
        root_dir="./datasets",
        transform=(None, None),
        max_bbox_shift=0,
        prompt_mode="box",
        min_points=1,
        max_points=1,
        image_size=256
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # 2. 建立 student 與 teacher
    student = build_model("configs/mobile_sam_finetune.yaml", None).cuda()
    teacher = build_model("configs/mobile_sam_orig.yaml", "weights/mobile_sam_orig.pth").cuda()
    teacher.eval()
    
    # 3. 取一個 batch 前向
    batch = next(iter(loader))
    imgs, masks, box, pts, lbls, is_pt = batch
    imgs = imgs.cuda(); masks = masks.cuda()

    # 4. student forward + loss 計算
    preds = student(imgs)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(preds, masks)
    assert bce.item() >= 0  # loss 非負
    
    # 5. 蒸餾 loss 計算
    #    直接呼叫 distill_losses 中的任何函式，用隨機數字測 shape
    from finetune_utils.distill_losses import encoder_matching_loss
    fake_s = [torch.randn(1, 64, 16, 16).cuda()]
    fake_t = [torch.randn(1, 64, 32, 32).cuda()]
    loss_enc = encoder_matching_loss(fake_s, fake_t)
    assert torch.isfinite(loss_enc).all()

    print("Smoke test passed: forward & loss OK.")

if __name__ == "__main__":
    smoke_run()
