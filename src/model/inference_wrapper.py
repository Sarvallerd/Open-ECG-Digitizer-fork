import torch
from src.utils import import_class_from_path
from typing import Dict, Tuple
from yacs.config import CfgNode as CN


class InferenceWrapper(torch.nn.Module):
    def __init__(
        self,
        config: CN,
        device: str,
        segment_then_resample: bool = True,
        resample_size: None | Tuple[int] = None,
        signal_class: int = 2,
    ) -> None:
        super(InferenceWrapper, self).__init__()
        self.config = config
        self.device = device
        self.segment_then_resample = segment_then_resample
        self.resample_size = resample_size
        self.signal_class = signal_class
        self.snake = self._load_snake()
        self.perspective_detector = self._load_perspective_detector()
        self.segmentation_model = self._load_segmentation_model().to(self.device)

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._check_image_dimensions(image)
        image = self.min_max_normalize(image)
        image = image.to(self.device)

        if self.segment_then_resample:
            signal_probabilities = self.segment_signal(image)
            image_aligned, src_points = self.perspective_detector(image)
            signal_probabilities_aligned = self.perspective_detector.apply_perspective(signal_probabilities, src_points)
            signal_probabilities_aligned = self.resample(signal_probabilities_aligned)
        else:  # resamples then semgents
            image = self.resample(image)
            image_aligned, src_points = self.perspective_detector(image)
            signal_probabilities_aligned = self.segment_signal(image_aligned)
        self.snake.fit(signal_probabilities_aligned.squeeze().cpu())

        out_dict = {
            "image": image.cpu(),
            "image_aligned": image_aligned.cpu(),
            "signal_probabilities_aligned": signal_probabilities_aligned.cpu(),
            "snake": self.snake.snake.data.detach(),
        }
        return out_dict

    def resample(self, image: torch.Tensor) -> torch.Tensor:
        if self.resample_size is not None:
            image = torch.nn.functional.interpolate(image, size=self.resample_size)
        return image

    def segment_signal(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.segmentation_model(image)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities[:, self.signal_class : self.signal_class + 1, :, :]

    def min_max_normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - image.min()) / (image.max() - image.min())

    def _load_snake(self) -> torch.nn.Module:
        snake_class = import_class_from_path(self.config.SNAKE.class_path)
        snake: torch.nn.Module = snake_class(**self.config.SNAKE.KWARGS)
        return snake

    def _load_perspective_detector(self) -> torch.nn.Module:
        perspective_detector_class = import_class_from_path(self.config.PERSPECTIVE_DETECTOR.class_path)
        perspective_detector: torch.nn.Module = perspective_detector_class(**self.config.PERSPECTIVE_DETECTOR.KWARGS)
        return perspective_detector

    def _load_segmentation_model(self) -> torch.nn.Module:
        segmentation_model_class = import_class_from_path(self.config.SEGMENTATION_MODEL.class_path)
        segmentation_model: torch.nn.Module = segmentation_model_class(**self.config.SEGMENTATION_MODEL.KWARGS)
        self._load_segmentation_model_weights(segmentation_model)
        return segmentation_model

    def _load_segmentation_model_weights(self, segmentation_model: torch.nn.Module) -> None:
        checkpoint = torch.load(
            self.config.SEGMENTATION_MODEL.weight_path, weights_only=True, map_location=self.device
        )[0]
        segmentation_model.load_state_dict(checkpoint)

    def _check_image_dimensions(self, image: torch.Tensor) -> None:
        if image.dim() != 4:
            raise NotImplementedError(f"Expected 4 dimensions, got tensor with {image.dim()} dimensions")
        if image.shape[0] != 1:
            raise NotImplementedError(f"Batch processing not supported, got tensor with shape {image.shape}")
        if image.shape[1] != 3:
            raise NotImplementedError(f"Expected 3 channels, got tensor with shape {image.shape}")
