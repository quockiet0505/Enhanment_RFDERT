"""Compatibility entrypoint: all new runs use the versioned train/test pipeline."""
import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from modal_app.modal_dqk_train_test import app, main
