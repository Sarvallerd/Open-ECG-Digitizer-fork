import numpy as np
import torch
import yaml
import torch.nn.functional as F
from torchvision.io import read_image
from torch import Tensor
from oed.model.unet import UNet
from oed.model.perspective_detector import PerspectiveDetector
from oed.model.cropper import Cropper
from oed.model.pixel_size_finder import PixelSizeFinder
from oed.model.signal_extractor import SignalExtractor
from oed.model.lead_identifier import LeadIdentifier


def add_noise_to_image(input_img, sigma=1.0, opacity=0.85):
    noise = torch.sigmoid(torch.randn_like(input_img) * sigma)
    input_img = (opacity) * input_img + (1 - opacity) * noise
    return input_img


def load_model(device: str, **kwargs):
    weights_path = kwargs.pop("weights_path", None)  # safely extract
    model = UNet(**kwargs)
    state_dict = torch.load(weights_path, map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model


def load_png_file(path):
    img = read_image(path)
    img = img.float() / 255.0
    img = img.unsqueeze(0)
    if img.shape[1] > 3:
        img = img[:, :3, :, :]
    return img


def _crop_y(
    image: Tensor,
    signal_prob: Tensor,
    grid_prob: Tensor,
    text_prob: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    def get_bounds(tensor: Tensor) -> tuple[int, int]:
        prob = torch.clamp(
            tensor.squeeze().sum(dim=tensor.dim() - 3)
            - tensor.squeeze().sum(dim=tensor.dim() - 3).mean(),
            min=0,
        )
        non_zero = (prob > 0).nonzero(as_tuple=True)[0]
        if non_zero.numel() == 0:
            return 0, tensor.shape[2] - 1
        return int(non_zero[0].item()), int(non_zero[-1].item())

    y1, y2 = get_bounds(signal_prob + grid_prob)

    slices = (slice(None), slice(None), slice(y1, y2 + 1), slice(None))
    return (
        image[slices],
        signal_prob[slices],
        grid_prob[slices],
        text_prob[slices],
    )


def _align_feature_maps(
    cropper: Cropper,
    image: Tensor,
    signal_prob: Tensor,
    grid_prob: Tensor,
    text_prob: Tensor,
    source_points: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    aligned_signal_prob = cropper.apply_perspective(
        signal_prob,
        source_points,
        fill_value=0,
    )
    aligned_image = cropper.apply_perspective(
        image,
        source_points,
        fill_value=0,
    )
    aligned_grid_prob = cropper.apply_perspective(
        grid_prob,
        source_points,
        fill_value=0,
    )
    aligned_text_prob = cropper.apply_perspective(
        text_prob,
        source_points,
        fill_value=0,
    )
    (
        aligned_image,
        aligned_signal_prob,
        aligned_grid_prob,
        aligned_text_prob,
    ) = _crop_y(
        aligned_image,
        aligned_signal_prob,
        aligned_grid_prob,
        aligned_text_prob,
    )

    return (
        aligned_image,
        aligned_signal_prob,
        aligned_grid_prob,
        aligned_text_prob,
    )


def plot_segmentation_and_image(
    image,
    segmentation,
    aligned_signal,
    aligned_grid,
    lines,
):
    import matplotlib.pyplot as plt

    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    probs = segmentation.squeeze(0).cpu()

    show_featuremap = torch.ones(probs.shape[1], probs.shape[2], 3)
    probs[2] /= probs[2].max()
    show_featuremap[:, :, [0, 1, 2]] -= 2 * probs[2].unsqueeze(-1)
    show_featuremap[:, :, [1, 2]] -= probs[0].unsqueeze(-1)
    show_featuremap = torch.clamp(show_featuremap, 0, 1).numpy()

    straightened_featuremap = torch.ones(
        aligned_signal.shape[2],
        aligned_signal.shape[3],
        3,
        device=aligned_signal.device,
    )
    aligned_signal /= aligned_signal.max()
    straightened_featuremap[:, :, [0, 1, 2]] -= 2 * aligned_signal[0, 0].unsqueeze(-1)
    aligned_grid /= aligned_grid.max()
    straightened_featuremap[:, :, [1, 2]] -= aligned_grid[0, 0].unsqueeze(-1)
    straightened_featuremap = torch.clamp(straightened_featuremap, 0, 1)

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    ax[0, 0].imshow(image_np)
    ax[0, 0].axis("off")

    ax[0, 1].imshow(show_featuremap)
    ax[0, 1].axis("off")

    ax[1, 0].imshow(straightened_featuremap.cpu())
    ax[1, 0].axis("off")

    offsets = [-0, -10.5, -7, -0, -3.5, -7, -0, -3.5, -7, -0, -3.5, -7]
    if lines.numel() > 0:
        ax[1, 1].plot(lines.T.cpu().numpy() + offsets[: lines.shape[0]])
    ax[1, 1].axis("off")
    plt.tight_layout()
    plt.show()


def crop_image(image, probs):
    perspective_detector = PerspectiveDetector(
        num_thetas=200,  # Higher -> more accurate but slower and more VRAM
        max_num_nonzero=10_000,
    )

    cropper = Cropper(
        granularity=50,
        percentiles=(0.03, 0.97),
        alpha=0.95,
    )

    alignment_params = perspective_detector(probs[0, 0])

    source_points = cropper(probs[0, 1], alignment_params)

    signal_prob, grid_prob, text_prob = (
        probs[:, [2]],
        probs[:, [0]],
        probs[:, [1]],
    )

    (
        aligned_image,
        aligned_signal_prob,
        aligned_grid_prob,
        aligned_text_prob,
    ) = _align_feature_maps(
        cropper,
        image,
        signal_prob,
        grid_prob,
        text_prob,
        source_points,
    )

    return (
        aligned_image,
        aligned_signal_prob,
        aligned_grid_prob,
        aligned_text_prob,
    )


def extract_signals(
    aligned_signal_prob: Tensor,
    aligned_grid_prob: Tensor,
    aligned_text_prob: Tensor,
    target_num_samples: int,
    pixel_size_finder_args: dict[str, int | float],
    signal_extractor_args: dict[str, int | float],
    device: str,
) -> Tensor:
    pixel_size_finder = PixelSizeFinder(**pixel_size_finder_args)
    signal_extractor = SignalExtractor(**signal_extractor_args)

    layout_unet = load_model(
        device=device,
        weights_path="/home/wozata/projects/Open-ECG-Digitizer-fork/weights/lead_name_unet_weights_07072025 copy.pt",
        num_in_channels=1,
        num_out_channels=13,
        dims=[32, 64, 128, 256, 256],
        depth=2,
    )

    layouts = yaml.safe_load(
        open(
            "/home/wozata/projects/Open-ECG-Digitizer-fork/oed/config/lead_layouts_george-moody-2024.yml",
            "r",
        ),
    )
    identifier = LeadIdentifier(
        layouts=layouts,
        unet=layout_unet,
        device=device,
        target_num_samples=target_num_samples,
    )
    mm_per_pixel_x, mm_per_pixel_y = pixel_size_finder(aligned_grid_prob)

    avg_pixel_per_mm = (
        1 / mm_per_pixel_x + 1 / mm_per_pixel_y
    ) / 2  # Is there a better way?
    signals = signal_extractor(aligned_signal_prob.squeeze())

    signals = identifier(
        signals,
        aligned_text_prob,
        avg_pixel_per_mm=avg_pixel_per_mm,
    )

    return signals


def resample_image(image: Tensor, resample_size: int | tuple[int, int]) -> Tensor:
    height, width = image.shape[2], image.shape[3]
    min_dim = min(height, width)
    max_dim = max(height, width)

    if isinstance(resample_size, int):
        if max_dim > resample_size:
            scale = resample_size / max_dim
            new_size = (int(height * scale), int(width * scale))
            return F.interpolate(
                image,
                size=new_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return image

    if isinstance(resample_size, tuple):
        interpolated = F.interpolate(
            image,
            size=resample_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return interpolated

    raise ValueError(
        f"Invalid resample_size: {resample_size}. Expected int or tuple of (height, width)."
    )


leads_names = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def get_slice(lead_name: str, number_of_rows: int) -> slice:
    assert lead_name in leads_names
    if lead_name in ("II",):
        return slice(0, number_of_rows)
    if lead_name in (("I", "III")):
        return slice(0, number_of_rows)
    if lead_name in (("aVR", "aVF", "aVL")):
        return slice(1 * number_of_rows, 2 * number_of_rows)
    if lead_name in (("V1", "V2", "V3")):
        return slice(2 * number_of_rows, 3 * number_of_rows)
    if lead_name in (("V4", "V5", "V6")):
        return slice(3 * number_of_rows, 4 * number_of_rows)


def digitize_image(
    input_img: Tensor,
    resample_size: int,
    target_num_samples: int,
    model: torch.nn.Module,
    device: str,
    pixel_size_finder_args: dict[str, int | float],
    signal_extractor_args: dict[str, int | float],
) -> Tensor:
    input_img = add_noise_to_image(input_img) # The UNet is trained for "real" images. Sometimes it performs better with added noise for generated images.
    input_img = resample_image(
        image=input_img, resample_size=resample_size
    )  # higher resample size is (probably) better but watch out for VRAM and time consraints

    with torch.no_grad():
        logits = model(input_img.to(device))
        output_probs = torch.softmax(logits, dim=1)
        aligned_image, aligned_signal, aligned_grid, aligned_text = crop_image(
            input_img, output_probs
        )
        lines = extract_signals(
            aligned_signal,
            aligned_grid,
            aligned_text,
            target_num_samples=target_num_samples,
            pixel_size_finder_args=pixel_size_finder_args,
            signal_extractor_args=signal_extractor_args,
            device=device,
        )
        lines = lines["canonical_lines"] * 1e-3  # microvolt to millivolt

    return output_probs, aligned_signal, aligned_grid, lines.float()


def calculate_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio in dB"""
    signal_power = np.mean(original**2)
    noise_power = np.mean((original - reconstructed) ** 2)

    if noise_power == 0:
        return np.inf
    if signal_power == 0:
        return -np.inf

    snr = 10 * np.log10(signal_power / noise_power)
    return snr
