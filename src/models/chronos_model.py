"""Chronos-2 Foundation Model Interface for Spatio-Temporal Traffic Forecasting.
=============================================================================
Provides the `ChronosForecaster` wrapper class around Amazon's Chronos-2 foundation
model (`amazon/chronos-2`), supporting:
1. Zero-shot foundation model inference across 307 PeMS04 sensors.
2. Loading and inference with fine-tuned LoRA parameter-efficient adapters.
3. Clean multivariate / per-sensor prediction interface preserving sensor index ordering.
4. Auto-detection of GPU (NVIDIA RTX 4060 with CUDA 12.1) or CPU fallback.
5. Inversion to physical engineering units (vehicles per 5 minutes).
"""

import os
# Prevent Keras 3 conflicts when transformers.trainer is imported
os.environ['USE_TF'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch

try:
    from chronos import Chronos2Pipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChronosForecaster:
    """Inference and adapter interface for Amazon Chronos-2 on highway traffic networks."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        base_model_id: str = "amazon/chronos-2",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """Initialize Chronos-2 forecaster.

        Args:
            checkpoint_path: Optional path to a fine-tuned LoRA checkpoint directory
                             (e.g., 'outputs/checkpoints/chronos2_lora/finetuned-ckpt').
                             If provided and valid, loads the fine-tuned adapter.
            base_model_id: HuggingFace hub model ID for base model (default: 'amazon/chronos-2').
            device: 'cuda', 'cpu', or None (auto-detect).
            torch_dtype: torch.bfloat16 or torch.float32 (auto-selected if None).
        """
        if not CHRONOS_AVAILABLE:
            raise ImportError(
                "chronos-forecasting is not installed. Install via `pip install chronos-forecasting peft`."
            )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if torch_dtype is None:
            if self.device == "cuda" and torch.cuda.is_available():
                # Ada Lovelace RTX 4060 natively accelerates bfloat16
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype

        self.base_model_id = base_model_id
        self.checkpoint_path = checkpoint_path
        self._is_fine_tuned = False
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self) -> Any:
        """Load the appropriate Chronos-2 pipeline (fine-tuned LoRA or zero-shot base)."""
        # 1. Try loading fine-tuned checkpoint if specified
        if self.checkpoint_path is not None:
            ckpt_path = Path(self.checkpoint_path)
            # Check for direct adapter files or finetuned-ckpt subfolder
            target_dir = None
            if (ckpt_path / "adapter_config.json").exists():
                target_dir = ckpt_path
            elif (ckpt_path / "finetuned-ckpt" / "adapter_config.json").exists():
                target_dir = ckpt_path / "finetuned-ckpt"

            if target_dir is not None:
                try:
                    logger.info(f"Loading fine-tuned Chronos-2 LoRA checkpoint from: {target_dir}")
                    pipeline = Chronos2Pipeline.from_pretrained(
                        str(target_dir),
                        device_map=self.device,
                        torch_dtype=self.torch_dtype,
                    )
                    self._is_fine_tuned = True
                    logger.info("Successfully loaded fine-tuned Chronos-2 LoRA model.")
                    return pipeline
                except Exception as e:
                    logger.warning(
                        f"Failed to load fine-tuned checkpoint from {target_dir}: {e}. "
                        f"Falling back to base pretrained model '{self.base_model_id}'."
                    )

        # 2. Load base pretrained model
        logger.info(f"Loading base Chronos-2 model: {self.base_model_id} on {self.device}")
        pipeline = Chronos2Pipeline.from_pretrained(
            self.base_model_id,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
        )
        self._is_fine_tuned = False
        return pipeline

    @property
    def is_fine_tuned(self) -> bool:
        """Return True if the loaded model is using a fine-tuned LoRA checkpoint."""
        return self._is_fine_tuned

    def predict_sensors(
        self,
        context_flow: Union[np.ndarray, torch.Tensor, List[torch.Tensor]],
        prediction_length: int = 12,
        batch_size: int = 512,
        is_normalized: bool = False,
        mean: float = 207.2275,
        std: float = 156.4778,
    ) -> np.ndarray:
        """Forecast future traffic flow for all 307 sensors, preserving sensor ordering.

        Args:
            context_flow: Historical flow observation. Can be:
                - 2D array of shape (context_length, 307)
                - 3D array of shape (batch_size, context_length, 307)
                - 3D array of shape (batch_size, 307, context_length)
                - List of 307 1D tensors
            prediction_length: Number of forecast steps ahead (default: 12 steps = 60 min).
            batch_size: Internal batch size for Chronos-2 generation.
            is_normalized: If True, input is z-score normalized and will be inverted.
            mean: Scaler mean for inverse normalization.
            std: Scaler std for inverse normalization.

        Returns:
            np.ndarray of shape (12, 307) for single sample,
            or (batch_size, 12, 307) for multiple samples,
            in physical engineering units (vehicles / 5 min).
        """
        # Convert torch to numpy for consistent shape handling
        if isinstance(context_flow, torch.Tensor):
            arr = context_flow.detach().cpu().numpy()
        elif isinstance(context_flow, list):
            arr = np.stack([x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x) for x in context_flow], axis=0)
            # arr shape is (307, context_length) -> convert to (1, 307, context_length)
            if arr.ndim == 2 and arr.shape[0] == 307:
                arr = arr[np.newaxis, ...]
        else:
            arr = np.array(context_flow)

        # Un-normalize if passed normalized values
        if is_normalized:
            arr = arr * std + mean

        single_sample = False

        # Format input tensor to (batch_size, 307, context_length)
        if arr.ndim == 2:
            # Shape is (context_length, 307) -> transpose to (1, 307, context_length)
            if arr.shape[1] == 307:
                arr = arr.T[np.newaxis, ...]
            elif arr.shape[0] == 307:
                arr = arr[np.newaxis, ...]
            single_sample = True
        elif arr.ndim == 3:
            # Could be (batch, context_length, 307) -> transpose to (batch, 307, context_length)
            if arr.shape[2] == 307 and arr.shape[1] != 307:
                arr = np.transpose(arr, (0, 2, 1))

        b_size, n_sensors, ctx_len = arr.shape
        assert n_sensors == 307, f"Expected 307 sensors, got {n_sensors}"

        # Chronos-2 accepts 3D tensor of shape (batch_size, n_variates, context_length)
        input_tensor = torch.from_numpy(arr).float()

        # Generate forecasts via Chronos2Pipeline
        # Returns (quantiles, mean) where mean has point forecasts
        quantiles, means = self.pipeline.predict_quantiles(
            input_tensor,
            prediction_length=prediction_length,
            batch_size=batch_size,
        )

        # means is a list of tensors of shape (307, prediction_length), one per batch element
        # We need (batch_size, prediction_length, 307)
        batch_preds = []
        for m in means:
            # m shape is (307, 12) -> transpose to (12, 307)
            pred_12_307 = m.cpu().float().numpy().T
            batch_preds.append(pred_12_307)

        output = np.stack(batch_preds, axis=0)  # Shape: (b_size, 12, 307)

        if single_sample:
            return output[0]  # Shape: (12, 307)
        return output

    def get_model_info(self) -> Dict[str, Any]:
        """Return architecture and deployment metadata."""
        n_params = sum(p.numel() for p in self.pipeline.model.parameters())
        trainable = sum(p.numel() for p in self.pipeline.model.parameters() if p.requires_grad)

        return {
            "model_type": "Chronos-2 (T5 Encoder Architecture)",
            "base_model_id": self.base_model_id,
            "is_fine_tuned": self._is_fine_tuned,
            "device": str(self.device),
            "torch_dtype": str(self.torch_dtype),
            "total_parameters": n_params,
            "trainable_parameters": trainable,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "context_length": self.pipeline.model_context_length,
            "prediction_length": self.pipeline.model_prediction_length,
        }
