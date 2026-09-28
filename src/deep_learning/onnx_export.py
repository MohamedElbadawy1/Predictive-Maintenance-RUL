"""
src/deep_learning/onnx_export.py — Keras 3 -> ONNX conversion, shared by
Pipeline/train_ensemble.py (which exports the real LSTM/GRU) and
tests/test_ensemble_inference_pipeline.py (which exports tiny throwaway
ones).

Lives in src/ rather than in Pipeline/ so nothing outside Pipeline/ has to
import from that folder: this project already had one bug where a
cross-folder import of Pipeline/ worked on some machines and broke on
others depending on the folder's exact case (see Sprint 19's doc), and
src/ is the one place imports are guaranteed to resolve everywhere.

Why a SavedModel export first, rather than tf2onnx's direct
`convert.from_keras(model, ...)`: as of tf2onnx 1.17 / Keras 3, the direct
path raises a KeyError from a stale tensor-name lookup against Keras 3's
renamed internals. Exporting a plain SavedModel and converting *that*
with the tf2onnx CLI sidesteps the incompatibility entirely.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from src.exceptions.custom_exception import CustomException
from src.logger.logger import logger


def export_to_onnx(model, out_path: Path, opset: int = 13) -> None:
    """Export a trained Keras model to `out_path` as ONNX."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_model_dir = Path(tmp_dir) / "saved_model"
        model.export(str(saved_model_dir))

        result = subprocess.run(
            [
                sys.executable, "-m", "tf2onnx.convert",
                "--saved-model", str(saved_model_dir),
                "--output", str(out_path),
                "--opset", str(opset),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise CustomException(
                f"tf2onnx failed converting {saved_model_dir} to {out_path}:\n{result.stderr}", sys,
            )
    logger.info(f"Exported ONNX model to {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
