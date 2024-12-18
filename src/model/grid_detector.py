from typing import Tuple, List, Dict
import torch
import torch.nn.functional as F


class GridDetector:

    def __init__(self, n_iter: int, num_thetas: int, smoothing_sigma: int):
        """
        Detects angles and distances using Hough Transform on the given image.

        Args:
            n_iter (int): The number of iterations for the Hough Transform,
                each iteration refines the detected angles and distances.
            num_thetas (int): The number of theta values to use in each iteration,
                reasonable values are > 50 and < 500.
            smoothing_sigma (int): The sigma value for Gaussian smoothing,
                in most cases, 1 will suffice.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                The detected vertical and horizontal angles (theta_vertical, theta_horizontal),
                and the mean distances (mean_distance_vertical, mean_distance_horizontal) between detected peaks.
        """
        self.n_iter = n_iter
        self.num_thetas = num_thetas
        self.smoothing_sigma = smoothing_sigma

    def create_gaussian_kernel(self, sigma: int) -> torch.Tensor:
        size = 5 * sigma
        x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
        y = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
        x, y = torch.meshgrid(x, y, indexing="ij")
        kernel = torch.exp(-0.5 * (x**2 + y**2) / sigma**2)
        kernel /= kernel.sum()
        return kernel.view(1, 1, size, size)

    def hough_transform(
        self, image: torch.Tensor, thetas: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies Hough Transform to detect lines in an image.

        Args:
            image (torch.Tensor): The input image tensor.
            thetas (torch.Tensor): The angles (thetas) for the Hough transform, in radians.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                The accumulator array, the rhos (radial distances), and the thetas, in radians.
        """
        device = image.device
        H, W = image.shape
        diag_len = int(torch.sqrt(torch.tensor(H**2 + W**2, device=device)))
        rhos = torch.linspace(-diag_len, diag_len, 2 * diag_len, device=device)
        num_thetas = len(thetas)
        num_rhos = len(rhos)

        y_idxs, x_idxs = torch.nonzero(image, as_tuple=True)
        cos_thetas = torch.cos(thetas)
        sin_thetas = torch.sin(thetas)

        x_idxs = x_idxs.view(-1, 1).float()
        y_idxs = y_idxs.view(-1, 1).float()
        rhos_vals = x_idxs * cos_thetas + y_idxs * sin_thetas
        rhos_idxs = torch.round((rhos_vals - rhos[0]) / (rhos[1] - rhos[0])).int()
        rhos_idxs = rhos_idxs.clamp(0, len(rhos) - 1)

        accumulator = torch.zeros(num_rhos * num_thetas, dtype=torch.int32, device=device)
        idxs_flat = rhos_idxs * num_thetas + torch.arange(num_thetas, device=device).reshape(1, -1)

        idxs_flat = idxs_flat.flatten()
        idxs_flat = idxs_flat[idxs_flat < num_rhos * num_thetas]

        accumulator.index_add_(0, idxs_flat, torch.ones_like(idxs_flat, dtype=torch.int32))
        accumulator = accumulator.view(len(rhos), len(thetas))

        return accumulator, rhos, thetas

    def detect_angles(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detects the angles iteratively using the Hough Transform.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                The vertical angle (theta_vertical),
                horizontal angle (theta_horizontal),
                the vertical FFT accumulator (fft_accumulator_vertical),
                and the horizontal FFT accumulator (fft_accumulator_horizontal).

                The FFT accumulators are the two columns represented by theta_vertical and theta_horizontal.
        """
        device = image.device
        thetas = torch.linspace(0, torch.pi, self.num_thetas, device=device) - torch.pi / 4

        kernel = self.create_gaussian_kernel(self.smoothing_sigma).to(device)

        for i in range(self.n_iter):
            accumulator, rhos, thetas = self.hough_transform(image, thetas)

            accumulator = (
                torch.nn.functional.conv2d(accumulator.float().unsqueeze(0).unsqueeze(0), kernel, padding="same")
                .squeeze(0)
                .squeeze(0)
            )

            fft_accumulator: torch.Tensor = torch.fft.rfft(accumulator, dim=0).abs()
            projected_accumulator = fft_accumulator.sum(0)

            hann = torch.hann_window(fft_accumulator.shape[1] // 2, device=device)
            hann = torch.cat([hann, hann])
            projected_accumulator *= hann

            n = len(projected_accumulator)
            first_peak = projected_accumulator[: n // 2].argmax()
            second_peak = projected_accumulator[n // 2 :].argmax() + n // 2

            thetas_vertical = torch.linspace(
                thetas[first_peak - 3], thetas[first_peak + 3], self.num_thetas // 2, device=device
            )
            thetas_horizontal = torch.linspace(
                thetas[second_peak - 3], thetas[second_peak + 3], self.num_thetas // 2, device=device
            )
            thetas = torch.cat([thetas_vertical, thetas_horizontal])

        return (
            thetas[first_peak].float(),
            thetas[second_peak].float(),
            fft_accumulator[:, first_peak],
            fft_accumulator[:, second_peak],
        )

    def __call__(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detects angles and distances (vertical and horizontal) from an image using Hough Transform.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                The vertical angle (theta_vertical),
                horizontal angle (theta_horizontal),
                the mean vertical distance (mean_distance_vertical),
                and the mean horizontal distance (mean_distance_horizontal).
        """
        theta_vertical, theta_horizontal, fft_accumulator_vertical, fft_accumulator_horizontal = self.detect_angles(
            image
        )
        accumulator_vertical = torch.fft.rfft(fft_accumulator_vertical, n=fft_accumulator_vertical.shape[0] * 2).abs()[
            :500
        ]
        accumulator_horizontal = torch.fft.rfft(
            fft_accumulator_horizontal, n=fft_accumulator_horizontal.shape[0] * 2
        ).abs()[:500]
        peaks_vertical = self.find_local_maxima(accumulator_vertical)
        peaks_horizontal = self.find_local_maxima(accumulator_horizontal)

        dist_vertical = torch.diff(torch.sort(peaks_vertical).values).float()
        dist_horizontal = torch.diff(torch.sort(peaks_horizontal).values).float()
        mean_dist_vertical = dist_vertical.mean()
        mean_dist_horizontal = dist_horizontal.mean()

        return theta_vertical, theta_horizontal, mean_dist_vertical, mean_dist_horizontal

    def find_local_maxima(self, tensor: torch.Tensor) -> torch.Tensor:
        padded_tensor = F.pad(tensor, (1, 1), value=float("-inf"))
        local_maxima = (padded_tensor[1:-1] > padded_tensor[:-2]) & (padded_tensor[1:-1] > padded_tensor[2:])
        indices = local_maxima.nonzero(as_tuple=True)[0]
        return indices


class MultiscaleGridDetector:

    def __init__(self, grid_detector: GridDetector, depth: int = 3, base: int = 5):
        """
        Detects angles and distances at multiple scales using the GridDetector.

        Args:
            grid_detector (GridDetector): The GridDetector instance.
            depth (int): The number of scales (default is 3).
            base (int): The base size for each scale (default is 5).

        Returns:
            Dict[str, List[torch.Tensor]]:
                The results for each scale, including vertical and horizontal angles (theta_vertical, theta_horizontal)
                and distances (mean_distance_vertical, mean_distance_horizontal).

        Example:
            depth = 3, base = 5
            will return results for scales 1, 5, and 25.
            for each parameter, tensors will be of shape (1, 1), (5, 5), and (25, 25).

            depth = 4, base = 2
            will return results for scales 1, 2, 4 and 8.
            for each parameter, tensors will be of shape (1, 1), (2, 2), (4, 4), and (8, 8).
        """
        self.grid_detector = grid_detector
        self.depth = depth
        self.grid_sizes = [base**i for i in range(depth)]

    @staticmethod
    def split_image(image: torch.Tensor, grid_size: int) -> torch.Tensor:
        """
        Splits an image into patches based on the grid size.

        Args:
            image (torch.Tensor): The input image tensor.
            grid_size (int): The grid size for patching the image.

        Returns:
            torch.Tensor: The patched image as a tensor.
        """
        height, width = image.shape[-2:]
        patch_height, patch_width = height // grid_size, width // grid_size
        patches = image.unfold(0, patch_height, patch_height).unfold(1, patch_width, patch_width)
        return patches.contiguous().view(-1, patch_height, patch_width)

    def __call__(self, image: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        device = image.device
        results: Dict[str, List[torch.Tensor]] = {
            "theta_vertical": [],
            "theta_horizontal": [],
            "distance_vertical": [],
            "distance_horizontal": [],
        }

        for grid_size in self.grid_sizes:
            results_per_scale = {key: torch.zeros((grid_size, grid_size), device=device) for key in results}

            patches = self.split_image(image, grid_size)

            for patch_idx, patch in enumerate(patches):
                row, col = divmod(patch_idx, grid_size)
                theta_vertical, theta_horizontal, distance_vertical, distance_horizontal = self.grid_detector(patch)

                results_per_scale["theta_vertical"][row, col] = theta_vertical
                results_per_scale["theta_horizontal"][row, col] = theta_horizontal
                results_per_scale["distance_vertical"][row, col] = distance_vertical
                results_per_scale["distance_horizontal"][row, col] = distance_horizontal

            for key in results:
                results[key].append(results_per_scale[key])

        return results
