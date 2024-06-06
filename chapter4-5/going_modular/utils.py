"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
from typing import Tuple

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
  
def export_model_to_explorer(model: torch.nn.Module,
                             sample_input: Tuple[torch.Tensor],
                             target_dir: str,
                             model_name: str):
  """
  like `save_model` function but this function will save model weights in `.pt2` format to visualize in ai-edge-model-explorer
  
  Args:
    model: A target PyTorch model to save.
    sample_input: A tuple of sample input data to be used for exporting the model.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
      
  Example usage:
    model = Unet(in_channels=3, out_channels=1)
    sample_input = torch.randn(1, 3, 224, 224)
    
    export_model_to_explorer(model=model, sample_input, target_dir="models", model_name="Unet_model")
  """
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  

  ex = torch.export.export(model, sample_input)
  torch.export.save(ex, target_dir_path / model_name + ".pt2")
