import torch
from pathlib import Path
from train_small_cnn import SmallCNN

def extract_state_dict():
    # Load the full model
    model_path = Path("data/raw/small_cnn.pth")
    
    # First create a new model instance
    model = SmallCNN()
    
    # Add the model class to safe globals and load with weights_only=True
    with torch.serialization.safe_globals([SmallCNN]):
        try:
            # Try loading with weights_only first
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception:
            # If that fails, try without weights_only
            loaded_model = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = loaded_model.state_dict()
    
    # Save just the state dict
    state_path = Path("data/raw/small_cnn_state.pth")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, state_path)
    print(f"Extracted state dict to {state_path}")

if __name__ == "__main__":
    extract_state_dict()