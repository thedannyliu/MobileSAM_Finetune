# finetune_utils/feature_hooks.py

import torch # 確保 torch 已導入

_FEATURE_STORE = {} # 全域變數用於儲存提取的特徵

def _get_hook(key: str):
    """
    為指定的鍵創建一個 hook 函數。
    """
    def fn(module, input, output): # hook 函數簽名是 (module, input, output)
        global _FEATURE_STORE
        feature_to_store = None
        if isinstance(output, torch.Tensor):
            feature_to_store = output.detach()
        elif isinstance(output, tuple) and len(output) > 0:
            # 如果輸出是元組，並且包含元素，則嘗試提取第一個元素
            # 這對於像 Transformer 這樣可能返回 (主要輸出, 其他信息) 的模組很常見
            # 例如，對於 mask_decoder.transformer，我們假設第一個元素 hs 是我們想要的
            if isinstance(output[0], torch.Tensor):
                feature_to_store = output[0].detach()
            else:
                print(f"Warning: Hook for '{key}', output was a tuple but its first element is not a Tensor. Type: {type(output[0])}")
        elif isinstance(output, list) and len(output) > 0: # 有些模組可能返回列表
            if isinstance(output[0], torch.Tensor):
                feature_to_store = output[0].detach()
            else:
                print(f"Warning: Hook for '{key}', output was a list but its first element is not a Tensor. Type: {type(output[0])}")
        else:
            print(f"Warning: Hook for '{key}', output type {type(output)} not handled or tuple/list is empty.")

        if feature_to_store is not None:
            _FEATURE_STORE.setdefault(key, []).append(feature_to_store)
        
    return fn

def register_hooks(model: torch.nn.Module, layer_names: list[str]):
    """
    為模型中指定的層註冊前向傳播的 hook。
    """
    global _FEATURE_STORE
    _FEATURE_STORE.clear() # 每次註冊新 hooks 時清空之前的特徵

    handles = []
    for name in layer_names:
        module_to_hook = model
        try:
            for part in name.split("."):
                module_to_hook = getattr(module_to_hook, part)
            handle = module_to_hook.register_forward_hook(_get_hook(name))
            handles.append(handle)
        except AttributeError:
            print(f"Warning: Layer '{name}' not found in model. Skipping hook registration for this layer.")
            
    return handles

def pop_features() -> dict:
    """
    獲取並清除儲存的特徵。
    返回一個字典，其中鍵是層名稱，值是包含捕獲到的特徵張量的列表（通常每個鍵的列表只包含一個張量，因為特徵在每次前向傳播後被 pop）。
    """
    global _FEATURE_STORE
    # 複製一份，因為 _FEATURE_STORE 會被清空
    # 並且確保返回值中的列表只包含實際的張量，而不是列表的列表（如果 _get_hook 邏輯改變）
    # 目前的 _get_hook 會 append 到 list, 所以 feats[key] 是一個 list of tensors.
    feats = dict(_FEATURE_STORE)
    _FEATURE_STORE.clear() # 為下一次 pop 做準備
    return feats