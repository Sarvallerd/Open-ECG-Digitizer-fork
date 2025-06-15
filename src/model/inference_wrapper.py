import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Tuple

import torch
import torch.nn.functional as F
from torch.nn import Module
from yacs.config import CfgNode as CN

from src.utils import import_class_from_path


@contextmanager
def timed_section(name: str, times_dict: Dict[str, float]) -> Generator[None, None, None]:
    start = time.time()
    yield
    times_dict[name] = time.time() - start


class InferenceWrapper(Module):
    def __init__(
        self,
        config: CN,
        device: str,
        resample_size: None | Tuple[int] = None,
        grid_class: int = 0,
        text_background_class: int = 1,
        signal_class: int = 2,
        background_class: int = 3,
        rotate_on_resample: bool = False,
        profile: bool = False,
        minimum_image_size: int = 512,
    ) -> None:
        super(InferenceWrapper, self).__init__()
        self.config = config
        self.device = device
        self.resample_size = resample_size
        self.grid_class = grid_class
        self.text_background_class = text_background_class
        self.signal_class = signal_class
        self.background_class = background_class
        self.rotate_on_resample = rotate_on_resample
        self._timing_enabled = profile
        self.snake = self._load_snake()
        self.perspective_detector: Any = self._load_perspective_detector()
        self.segmentation_model: Any = self._load_segmentation_model().to(self.device)
        self.cropper: Any = self._load_cropper()
        self.pixel_size_finder: Any = self._load_pixel_size_finder()
        self.dewarper: Any = self._load_dewarper()
        self.minimum_image_size = minimum_image_size
        self.times: dict[str, float] = {}

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._check_image_dimensions(image)
        image = self.min_max_normalize(image)
        image = image.to(self.device)

        self.times = {}
        image = self._resample_image(image)

        signal_prob, grid_prob = self._get_feature_maps(image)

        with timed_section("Perspective detection", self.times):
            alignment_params = self.perspective_detector(grid_prob)

        with timed_section("Cropping", self.times):
            source_points = self.cropper(signal_prob, alignment_params)

        aligned_image, aligned_signal_prob, aligned_grid_prob = self._align_feature_maps(
            image, signal_prob, grid_prob, source_points
        )

        with timed_section("Pixel size search", self.times):
            mm_per_pixel_x, mm_per_pixel_y = self.pixel_size_finder(aligned_grid_prob)

        with timed_section("Dewarping", self.times):
            avg_pixel_per_mm = (1 / mm_per_pixel_x + 1 / mm_per_pixel_y) / 2
            self.dewarper.fit(aligned_grid_prob.squeeze(), avg_pixel_per_mm)
            aligned_signal_prob = self.dewarper.transform(aligned_signal_prob.squeeze())

        with timed_section("Signal extraction", self.times):
            self.snake.fit(aligned_signal_prob.squeeze().cpu())

        self._print_profiling_results()

        return {
            "image": image.cpu(),
            "image_aligned": aligned_image.cpu(),
            "signal_probabilities_aligned": aligned_signal_prob.cpu(),
            "grid_probabilities_aligned": aligned_grid_prob.cpu(),
            "snake": self.snake.snake.data.detach(),
            "mm_per_pixel_x": mm_per_pixel_x,
            "mm_per_pixel_y": mm_per_pixel_y,
            "source_points": source_points.cpu(),
        }

    def _align_feature_maps(
        self, image: torch.Tensor, signal_prob: torch.Tensor, grid_prob: torch.Tensor, source_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with timed_section("Feature map resampling", self.times):
            aligned_signal_prob = self.cropper.apply_perspective(signal_prob, source_points, fill_value=0)
            aligned_image = self.cropper.apply_perspective(image, source_points, fill_value=0)
            aligned_grid_prob = self.cropper.apply_perspective(grid_prob, source_points, fill_value=0)
            if self.rotate_on_resample:
                aligned_image, aligned_signal_prob, aligned_grid_prob = self._rotate_on_resample(
                    aligned_image, aligned_signal_prob, aligned_grid_prob
                )
            aligned_image, aligned_signal_prob, aligned_grid_prob = self._crop_y(
                aligned_image,
                aligned_signal_prob,
                aligned_grid_prob,
            )

            return aligned_image, aligned_signal_prob, aligned_grid_prob

    def _crop_y(
        self, image: torch.Tensor, signal_prob: torch.Tensor, grid_prob: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def get_bounds(tensor: torch.Tensor) -> Tuple[int, int]:
            prob = torch.clamp(
                (tensor).squeeze().sum(dim=tensor.dim() - 3) - (tensor).squeeze().sum(dim=tensor.dim() - 3).mean(),
                min=0,
            )
            non_zero = (prob > 0).nonzero(as_tuple=True)[0]
            return int(non_zero[0].item()), int(non_zero[-1].item())

        y1, y2 = get_bounds(signal_prob + grid_prob)
        x1, x2 = get_bounds((signal_prob + grid_prob).transpose(-2, -1))

        slices = (slice(None), slice(None), slice(y1, y2 + 1), slice(x1, x2 + 1))
        return image[slices], signal_prob[slices], grid_prob[slices]

    def _print_profiling_results(self) -> None:
        if not self._timing_enabled:
            return
        print("Profiling times")
        max_length = max(len(section) for section in self.times.keys())
        for section, duration in self.times.items():
            print(f"    {section:<{max_length+2}}{duration:.2f}s")
        total_time = sum(self.times.values())
        print(f"Total time: {total_time:.2f}s")

    def _rotate_on_resample(
        self, aligned_image: torch.Tensor, aligned_signal_prob: torch.Tensor, aligned_grid_prob: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if aligned_image.shape[2] > aligned_image.shape[3]:
            aligned_image = torch.rot90(aligned_image, k=1, dims=(2, 3))
            aligned_signal_prob = torch.rot90(aligned_signal_prob, k=1, dims=(2, 3))
            aligned_grid_prob = torch.rot90(aligned_grid_prob, k=1, dims=(2, 3))
        return aligned_image, aligned_signal_prob, aligned_grid_prob

    def _resample_image(self, image: torch.Tensor) -> torch.Tensor:
        with timed_section("Initial resampling", self.times):
            if self.resample_size is None:
                return image

            height, width = image.shape[2], image.shape[3]
            min_dim = min(height, width)
            max_dim = max(height, width)

            # Upsample if image is smaller than minimum allowed size
            if min_dim < self.minimum_image_size:
                scale: float = self.minimum_image_size / min_dim
                new_size: tuple[int, int] = (int(height * scale), int(width * scale))
                interpolated: torch.Tensor = F.interpolate(image, size=new_size, mode="bilinear", align_corners=False)
                return interpolated

            # Downsample if resample_size is an int and image is larger
            if isinstance(self.resample_size, int):
                if max_dim > self.resample_size:
                    scale = self.resample_size / max_dim
                    new_size = (int(height * scale), int(width * scale))
                    return F.interpolate(image, size=new_size, mode="bilinear", align_corners=False)
                return image

            # Resize directly if resample_size is a tuple
            if isinstance(self.resample_size, tuple):
                interpolated = F.interpolate(image, size=self.resample_size, mode="bilinear", align_corners=False)
                return interpolated

            raise ValueError(f"Invalid resample_size: {self.resample_size}. Expected int or tuple of (height, width).")

    def process_sparse_prob(self, signal_prob: torch.Tensor) -> torch.Tensor:
        signal_prob = signal_prob - signal_prob.mean() * 1
        signal_prob = torch.clamp(signal_prob, min=0)
        signal_prob = signal_prob / (signal_prob.max() + 1e-9)
        return signal_prob

    def _get_feature_maps(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with timed_section("Segmentation", self.times):
            logits = self.segmentation_model(image)
            prob = torch.softmax(logits, dim=1)

            signal_prob = prob[:, [self.signal_class], :, :]
            grid_prob = prob[:, [self.grid_class], :, :]

            signal_prob = self.process_sparse_prob(signal_prob)
            grid_prob = self.process_sparse_prob(grid_prob)

            return signal_prob, grid_prob

    def min_max_normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - image.min()) / (image.max() - image.min())

    def _load_snake(self) -> Any:
        snake_class = import_class_from_path(self.config.SNAKE.class_path)
        snake: Any = snake_class(**self.config.SNAKE.KWARGS)
        return snake

    def _load_perspective_detector(self) -> Any:
        perspective_detector_class = import_class_from_path(self.config.PERSPECTIVE_DETECTOR.class_path)
        perspective_detector: Any = perspective_detector_class(**self.config.PERSPECTIVE_DETECTOR.KWARGS)
        return perspective_detector

    def _load_segmentation_model(self) -> Any:
        segmentation_model_class = import_class_from_path(self.config.SEGMENTATION_MODEL.class_path)
        segmentation_model: Any = segmentation_model_class(**self.config.SEGMENTATION_MODEL.KWARGS)
        self._load_segmentation_model_weights(segmentation_model)
        return segmentation_model.eval()

    def _load_cropper(self) -> Any:
        cropper_class = import_class_from_path(self.config.CROPPER.class_path)
        cropper: Any = cropper_class(**self.config.CROPPER.KWARGS)
        return cropper

    def _load_pixel_size_finder(self) -> Any:
        pixel_size_finder_class = import_class_from_path(self.config.PIXEL_SIZE_FINDER.class_path)
        pixel_size_finder: Any = pixel_size_finder_class(**self.config.PIXEL_SIZE_FINDER.KWARGS)
        return pixel_size_finder

    def _load_dewarper(self) -> Any:
        dewarper_class = import_class_from_path(self.config.DEWARPER.class_path)
        dewarper: Any = dewarper_class(**self.config.DEWARPER.KWARGS)
        return dewarper

    def _load_segmentation_model_weights(self, segmentation_model: torch.nn.Module) -> None:
        checkpoint = torch.load(self.config.SEGMENTATION_MODEL.weight_path, weights_only=True, map_location=self.device)
        if isinstance(checkpoint, tuple):
            checkpoint = checkpoint[0]
        checkpoint = {
            k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()
        }  # torch.compile adds _orig_mod. to keys
        segmentation_model.load_state_dict(checkpoint)

    def _check_image_dimensions(self, image: torch.Tensor) -> None:
        if image.dim() != 4:
            raise NotImplementedError(f"Expected 4 dimensions, got tensor with {image.dim()} dimensions")
        if image.shape[0] != 1:
            raise NotImplementedError(f"Batch processing not supported, got tensor with shape {image.shape}")
        if image.shape[1] != 3:
            raise NotImplementedError(f"Expected 3 channels, got tensor with shape {image.shape}")
