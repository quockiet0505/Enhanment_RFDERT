"""Export the exact validation winner with the metadata stripped by RF-DETR."""
from pathlib import Path
import torch


def export_best(output, model_config, names, notes):
    output = Path(output)
    winner = torch.load(output/"checkpoint_best_total.pth", map_location="cpu", weights_only=True)
    source = winner.get("best_total_source")
    if source not in {"ema", "regular"}:
        raise ValueError("Missing best-total winner provenance")
    original = torch.load(output/f"checkpoint_best_{source}.pth", map_location="cpu", weights_only=True)
    left, right = winner["model"], original["model"]
    if set(left) != set(right) or any(not torch.equal(left[k], right[k]) for k in left):
        raise ValueError("Winner weights differ from the recorded source")
    # Use the winner's actual epoch, never the last epoch or trainer.current_epoch.
    winner["epoch"] = original["epoch"]
    winner["model_config"] = model_config
    winner["args"]["class_names"] = list(names)
    winner["args"]["notes"] = notes
    winner["dqk_best_source"] = str(output/f"checkpoint_best_{source}.pth")
    temporary = output/"best.pending.pth"
    torch.save(winner, temporary)
    temporary.replace(output/"best.pth")
    return str(output/"best.pth")
