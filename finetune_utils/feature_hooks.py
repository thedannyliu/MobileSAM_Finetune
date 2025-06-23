# finetune_utils/feature_hooks.py
import torch

_FEATURE_STORE = {}  # 全域變數用於儲存提取的特徵


def _get_hook(key: str):
    """
    為指定的鍵創建一個 hook 函數。
    """

    def fn(module, input, output):  # PyTorch hook 的標準簽名
        global _FEATURE_STORE
        feature_to_store = None

        if isinstance(output, torch.Tensor):
            feature_to_store = output.detach().clone()  # Detach and clone
        elif isinstance(output, tuple) and len(output) > 0:
            # 如果輸出是元組，我們主要關心第一個元素（通常是主要的特徵張量，例如 hs）
            if isinstance(output[0], torch.Tensor):
                feature_to_store = output[0].detach().clone()  # Detach and clone the first tensor
            else:
                print(
                    f"Warning: Hook for '{key}': Output was a tuple, but its first element is not a Tensor. Type: {type(output[0])}. Storing as is if other elements are tensors, or skipping."
                )
                # 可選：如果希望儲存元組中所有的張量，或者將它們分別儲存
                # Example: Store all tensors in the tuple under the same key as a list of tensors
                # temp_list = []
                # for item in output:
                # if isinstance(item, torch.Tensor):
                # temp_list.append(item.detach().clone())
                # if temp_list:
                # feature_to_store = temp_list # Will store a list of tensors
                # else:
                # print(f"Warning: Hook for '{key}': Output tuple contained no tensors.")
        elif isinstance(output, list) and len(output) > 0:
            if isinstance(output[0], torch.Tensor):
                feature_to_store = output[0].detach().clone()  # Detach and clone the first tensor
            else:
                print(
                    f"Warning: Hook for '{key}': Output was a list, but its first element is not a Tensor. Type: {type(output[0])}."
                )
        else:
            if output is not None:  # 避免 NoneType 錯誤
                print(
                    f"Warning: Hook for '{key}': Output type {type(output)} is not explicitly handled for detach. Storing as is if possible or skipping."
                )
            # 对于未处理的类型或 None，feature_to_store 保持 None

        if feature_to_store is not None:
            # 如果_FEATURE_STORE[key]不存在，setdefault會創建一個空列表並返回它，然後append。
            # 如果已存在，直接append。
            _FEATURE_STORE.setdefault(key, []).append(feature_to_store)
        elif output is not None and not isinstance(output, (torch.Tensor, tuple, list)):
            # 如果是一些其他類型的單一物件，且不是None，也許可以直接儲存（如果下游客戶端能處理）
            # 但通常我們只關心 Tensor。為安全起見，只儲存已知的可分離張量。
            print(
                f"Info: Hook for '{key}', output type {type(output)} was not a tensor or handled tuple/list, not stored."
            )

    return fn


def _get_pre_hook(key: str):
    def fn(module, input):
        global _FEATURE_STORE
        x = input[0]
        if isinstance(x, torch.Tensor):
            _FEATURE_STORE.setdefault(key, []).append(x.detach().clone())

    return fn


def register_hooks(model: torch.nn.Module, layer_names: list[str]):
    """
    為模型中指定的層註冊前向傳播的 hook。
    """
    global _FEATURE_STORE
    _FEATURE_STORE.clear()  # 每次註冊新 hooks 時清空之前的特徵，確保每個 split 或模型獨立

    handles = []
    for name in layer_names:
        pre = False
        clean = name
        if name.startswith("!"):
            pre = True
            clean = name[1:]
        module_to_hook = model
        try:
            for part in clean.split("."):
                module_to_hook = getattr(module_to_hook, part)
            if pre:
                handle = module_to_hook.register_forward_pre_hook(_get_pre_hook(clean))
            else:
                handle = module_to_hook.register_forward_hook(_get_hook(clean))
            handles.append(handle)
        except AttributeError:
            print(
                f"Warning: Layer '{clean}' not found in model. Skipping hook registration for this layer."
            )

    return handles


def pop_features() -> dict:
    """
    獲取並清除儲存的特徵。
    返回一個字典，其中鍵是層名稱，值是包含捕獲到的特徵張量的列表。
    由於我們通常在每次 forward pass 後調用 pop_features，所以這個列表預期只包含一個元素。
    """
    global _FEATURE_STORE
    feats = dict(_FEATURE_STORE)  # 創建副本
    _FEATURE_STORE.clear()  # 清空全域儲存，為下一次 forward pass 做準備
    return feats
