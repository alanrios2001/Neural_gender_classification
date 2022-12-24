import torch
from pathlib import Path
import json
import time


def save_model(model: torch.nn.Module,
               stats: dict,
               target_dir: str,
               model_name: str):

    current_time = time.strftime("%H%M%S")
    full_path = target_dir + model_name.split(".")[0] + current_time
    model_path = Path(full_path)
    model_path.mkdir(parents=True, exist_ok=True)

    model_save_path = model_path / model_name

    torch.save(obj=model.state_dict(), f=model_save_path)
    with open(f"{full_path}/metrics.json", "a") as json_file:
        json.dump(stats, json_file, indent=4)

    print(f"model saved in {str(model_save_path)}")
