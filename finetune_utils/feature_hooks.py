# finetune_utils/feature_hooks.py
import torch

_FEATURE_STORE = {} # Global variable to store extracted features

def _get_hook(key: str):
    """
    Creates a hook function for the specified key.
    """
    def fn(module, input, output): # Standard signature for a PyTorch hook
        global _FEATURE_STORE
        feature_to_store = None
        
        if isinstance(output, torch.Tensor):
            feature_to_store = output.detach().clone() # Detach and clone
        elif isinstance(output, tuple) and len(output) > 0:
            # If the output is a tuple, we are mainly interested in the first element
            # (usually the main feature tensor, e.g., hs).
            if isinstance(output[0], torch.Tensor):
                feature_to_store = output[0].detach().clone() # Detach and clone the first tensor
            else:
                print(f"Warning: Hook for '{key}': Output was a tuple, but its first element is not a Tensor. Type: {type(output[0])}. Storing as is if other elements are tensors, or skipping.")
                # Optional: If you want to store all tensors in the tuple, or store them separately
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
                feature_to_store = output[0].detach().clone() # Detach and clone the first tensor
            else:
                print(f"Warning: Hook for '{key}': Output was a list, but its first element is not a Tensor. Type: {type(output[0])}.")
        else:
            if output is not None: # Avoid NoneType error
                print(f"Warning: Hook for '{key}': Output type {type(output)} is not explicitly handled for detach. Storing as is if possible or skipping.")
            # For unhandled types or None, feature_to_store remains None

        if feature_to_store is not None:
            # If _FEATURE_STORE[key] does not exist, setdefault creates an empty list and returns it, then appends.
            # If it already exists, it just appends.
            _FEATURE_STORE.setdefault(key, []).append(feature_to_store)
        elif output is not None and not isinstance(output, (torch.Tensor, tuple, list)):
             # If it's some other type of single object and not None, maybe it can be stored directly
             # (if the downstream client can handle it). But usually we only care about Tensors.
             # To be safe, only store known detachable tensors.
            print(f"Info: Hook for '{key}', output type {type(output)} was not a tensor or handled tuple/list, not stored.")
            
    return fn

def register_hooks(model: torch.nn.Module, layer_names: list[str]):
    """
    Registers forward hooks for the specified layers in the model.
    """
    global _FEATURE_STORE
    _FEATURE_STORE.clear() # Clear previous features each time new hooks are registered to ensure each split or model is independent

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
    Retrieves and clears the stored features.
    Returns a dictionary where keys are layer names and values are lists
    containing the captured feature tensors. Since we typically call pop_features
    after each forward pass, this list is expected to contain only one element.
    """
    global _FEATURE_STORE
    feats = dict(_FEATURE_STORE) # Create a copy
    _FEATURE_STORE.clear()       # Clear the global store to prepare for the next forward pass
    return feats