from typing import Tuple, Dict
from torchvision.transforms.functional import perspective
from scipy.ndimage import uniform_filter
from skimage.filters import threshold_multiotsu
from skimage.morphology import remove_small_objects, skeletonize
from torch.nn.functional import avg_pool2d
import torch
import matplotlib.pyplot as plt

DEBUG = False


class PerspectiveDetector:

    def __init__(self, num_thetas: int, percentile: float = 0.01):
        self.num_thetas = num_thetas
        self.percentile = percentile

    def hough_transform(self, image: torch.Tensor, thetas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies Hough Transform to detect lines in an image.

        Args:
            image (torch.Tensor): The input image tensor.
            thetas (torch.Tensor): The angles (thetas) for the Hough transform, in radians.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The accumulator array, the rhos (radial distances).
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

        # Remove most of the negative rhos
        rhos = rhos[num_rhos // 3 :]
        accumulator = accumulator[num_rhos // 3 :]

        return accumulator, rhos

    def get_line_values(
        self, accumulator: torch.Tensor, thetas: torch.Tensor, rhos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Extract the values along a line in the Hough accumulator, parameterized by two thetas: one at the top and
        one at the bottom. The thetas are chosen to maximize the variance along the line and correspond to the
        angles of the lines in the image.

        Args:
            accumulator (torch.Tensor): The Hough accumulator tensor.
            thetas (torch.Tensor): The angles (thetas) corresponding to dim 1 of the tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The values along the line, the corresponding thetas,
                and the radial distances.
        """
        variance = self.calculate_line_variances(accumulator)
        max_idx_top = variance.argmax() // (self.num_thetas // 2)
        max_idx_bottom = variance.argmax() % (self.num_thetas // 2)

        line_values, line_thetas = self.extract_line(
            accumulator,
            thetas,
            max_idx_top,
            max_idx_bottom,
        )

        line_cumsum = line_values.cumsum(dim=0) / line_values.sum()
        start = (line_cumsum > self.percentile).nonzero(as_tuple=True)[0][0]
        end = (line_cumsum > 1 - self.percentile).nonzero(as_tuple=True)[0][0]

        return rhos[start:end], line_thetas[start:end], line_values[start:end]

    def get_theta_lims(self, accumulator: torch.Tensor, thetas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the variance along lines in the Hough accumulator, parameterized by two thetas: one at the top and
        one at the bottom. The thetas are chosen to maximize the variance along the line and correspond to the
        angles of the lines in the image.

        Args:
            accumulator (torch.Tensor): The Hough accumulator tensor.
            thetas (torch.Tensor): The angles (thetas) corresponding to dim 1 of the tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The thetas corresponding to the top and bottom of the accumulator.
        """
        variance = self.calculate_line_variances(accumulator)
        max_idx_top = variance.argmax() // (self.num_thetas // 2)
        max_idx_bottom = variance.argmax() % (self.num_thetas // 2)

        theta_top = thetas[max_idx_top]  # index at top of accumulator
        theta_bottom = thetas[max_idx_bottom]  # index at bottom of accumulator

        theta_min = torch.min(theta_top, theta_bottom)
        theta_max = torch.max(theta_top, theta_bottom)

        return theta_min, theta_max

    def find_bounding_lines(self, image: torch.Tensor, eps: float) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Find the bounding lines of the gridded paper in an image taken from a camera.

        Args:
            image (torch.Tensor): The input image tensor, with shape [H, W]. Should be a binary image.
            eps (float): Margin added to the thetas for the second pass of the Hough transform.

        Returns:
            Dict[str, Tuple[torch.Tensor, torch.Tensor]]: The radial and angular values of the bounding lines.
        """
        hann = (
            torch.cat([torch.hann_window(self.num_thetas // 2), torch.hann_window(self.num_thetas // 2)])
            .unsqueeze(0)
            .to(image.device)
        )

        # PASS 1
        thetas = self.get_initial_thetas().to(image.device)
        accumulator, rhos = self.hough_transform(image, thetas)
        accumulator = accumulator * hann

        if DEBUG:
            fig, ax = plt.subplots(1, 1, figsize=(25, 15))
            ax.imshow(accumulator.cpu().numpy(), aspect="auto", cmap="nipy_spectral", interpolation="nearest")
            plt.tight_layout()
            plt.savefig("accumulator1.png")
            plt.show()

        (
            theta_min_horizontal,
            theta_max_horizontal,
        ) = self.get_theta_lims(accumulator[:, : self.num_thetas // 2], thetas[: self.num_thetas // 2])
        (
            theta_min_vertical,
            theta_max_vertical,
        ) = self.get_theta_lims(accumulator[:, self.num_thetas // 2 :], thetas[self.num_thetas // 2 :])

        # PASS 2
        thetas = torch.cat(
            [
                torch.linspace(
                    theta_min_horizontal - eps,
                    theta_max_horizontal + eps,
                    self.num_thetas // 2,
                    device=image.device,
                ),
                torch.linspace(
                    theta_min_vertical - eps,
                    theta_max_vertical + eps,
                    self.num_thetas // 2,
                    device=image.device,
                ),
            ]
        )
        accumulator, rhos = self.hough_transform(image, thetas)
        accumulator = accumulator * hann

        if DEBUG:
            fig, ax = plt.subplots(1, 1, figsize=(25, 15))
            ax.imshow(accumulator.cpu().numpy(), aspect="auto", cmap="nipy_spectral", interpolation="nearest")
            plt.tight_layout()
            plt.savefig("accumulator2.png")
            plt.show()

        (
            rhos_horizontal,
            thetas_horizontal,
            values_horizontal,
        ) = self.get_line_values(accumulator[:, : self.num_thetas // 2], thetas[: self.num_thetas // 2], rhos)
        (
            rhos_vertical,
            thetas_vertical,
            values_vertical,
        ) = self.get_line_values(accumulator[:, self.num_thetas // 2 :], thetas[self.num_thetas // 2 :], rhos)

        if DEBUG:
            fig, ax = plt.subplots(1, 1, figsize=(25, 15))
            ax.plot(values_horizontal.cpu().numpy(), label="Horizontal")
            ax.plot(values_vertical.cpu().numpy(), label="Vertical")
            ax.legend()
            plt.show()

        params = {
            "left": (rhos_horizontal[0], thetas_horizontal[0]),
            "right": (rhos_horizontal[-1], thetas_horizontal[-1]),
            "top": (rhos_vertical[0], thetas_vertical[0]),
            "bottom": (rhos_vertical[-1], thetas_vertical[-1]),
        }

        return params

    def get_initial_thetas(self) -> torch.Tensor:
        return torch.cat(
            [
                torch.linspace(-torch.pi / 4, torch.pi / 4, self.num_thetas // 2),
                torch.linspace(torch.pi / 4, 3 * torch.pi / 4, self.num_thetas // 2),
            ]
        )

    def __call__(self, image: torch.Tensor, eps: float = torch.pi / 180) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Correct the perspective of an image of a gridded paper.

        Args:
            image (torch.Tensor): The input image tensor, with shape [C, H, W] or [H,W].
            eps (float): Margin added to the thetas for the second pass of the Hough transform.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The corrected image tensor and the source points in pixel coordiantes.

        """
        binary_image = self.binarize(image)

        if DEBUG:
            fig, ax = plt.subplots(1, 1, figsize=(25, 15))
            ax.imshow(binary_image.cpu().numpy(), cmap="gray")
            plt.tight_layout()
            plt.savefig("binary_image.png")
            plt.show()

        params = self.find_bounding_lines(binary_image, eps)

        source_points = torch.stack(
            [
                self.line_intersection_from_hough(*params["top"], *params["left"]),
                self.line_intersection_from_hough(*params["top"], *params["right"]),
                self.line_intersection_from_hough(*params["bottom"], *params["right"]),
                self.line_intersection_from_hough(*params["bottom"], *params["left"]),
            ]
        )

        return self.apply_perspective(image, source_points), source_points

    def apply_perspective(self, image: torch.Tensor, source_points: torch.Tensor) -> torch.Tensor:
        H, W = image.shape[-2:]
        destination_points = [[0.0, 0.0], [W, 0.0], [W, H], [0.0, H]]

        corrected: torch.Tensor = perspective(image.unsqueeze(0), source_points.tolist(), destination_points).squeeze(0)

        return corrected

    def quantile(self, tensor: torch.Tensor, q: float, max_num_elems: int = 100_000) -> torch.Tensor:
        r"""
        Compute the q-th quantile of a tensor.

        Args:
            tensor (torch.Tensor): The input tensor.
            q (float): The quantile to compute.

        Returns:
            torch.Tensor: The q-th quantile of the tensor.
        """
        flattened_tensor = tensor.flatten()
        if len(flattened_tensor) > max_num_elems:
            indices = torch.randperm(len(flattened_tensor))[:max_num_elems]
            flattened_tensor = flattened_tensor[indices]
        return torch.quantile(flattened_tensor, q)

    def binarize(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalizes image to compensate for uneven lighting, then binarizes image by applying a two-sided threshold.
        Lastly, removes clutter and skeletonizes the image to preserve grid-like structures.

        Args:
            image (torch.Tensor): The input image tensor, with shape [C, H, W] or [H, W].

        Returns:
            torch.Tensor: The binarized image tensor with shape [H, W].
        """
        device = image.device
        if image.ndim == 3:
            flat_image = image.float().mean(0)
        else:
            flat_image = image.float()

        blurred = avg_pool2d(flat_image.unsqueeze(0).unsqueeze(0), 9, stride=1, padding=4).squeeze()
        differenced = flat_image - blurred

        local_minima_candidates: torch.Tensor = differenced < self.quantile(differenced, 0.1)

        local_minima_candidates_np = remove_small_objects(
            local_minima_candidates.cpu().numpy(), min_size=64, connectivity=2
        )  # type: ignore
        local_minima_candidates = torch.tensor(local_minima_candidates_np, dtype=torch.bool).to(device)

        flat_image_np = flat_image.detach().cpu().numpy()
        flat_image_np = flat_image_np / uniform_filter(flat_image_np, size=35)
        thresholds = threshold_multiotsu(flat_image_np, 4)  # type: ignore
        flat_image_np = (flat_image_np > thresholds[0]) & (flat_image_np < thresholds[1])
        flat_image_np = remove_small_objects(flat_image_np, min_size=16, connectivity=2)  # type: ignore
        flat_image_np = skeletonize(flat_image_np)  # type: ignore
        binary_image: torch.Tensor = torch.tensor(flat_image_np, dtype=torch.bool).to(device)
        return binary_image & local_minima_candidates

    def calculate_line_variances(self, accumulator: torch.Tensor) -> torch.Tensor:
        """
        Calculate variance along the derivative for all lines from top to bottom of an image.

        Quadratic complexity in the number of thetas (W).

        Args:
            accumulator (torch.Tensor): The Hough accumulator tensor.

        Returns:
            torch.Tensor: The variance tensor with shape [W, W].
        """
        H, W = accumulator.shape
        x_start, x_end = torch.meshgrid(torch.arange(W), torch.arange(W), indexing="ij")
        slopes = torch.where(x_start != x_end, H / (x_end - x_start).float(), torch.tensor(float("inf")))
        y_coords = torch.arange(H).view(-1, 1, 1).float()
        x_coords = torch.where(
            slopes == float("inf"), x_start.float(), x_start + y_coords / slopes  # Diagonal entries: x_start == x_end
        )
        x_coords_clamped = torch.clamp(x_coords.round(), 0, W - 1).long()
        sampled_values = accumulator[y_coords.long(), x_coords_clamped]
        derivatives = torch.diff(sampled_values, dim=0)
        variances = torch.var(derivatives, dim=0)
        return variances

    def extract_line(
        self, accumulator: torch.Tensor, thetas: torch.Tensor, x_start: torch.Tensor, x_end: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Extract a line from a Hough accumulator tensor.

        Args:
            accumulator (torch.Tensor): The Hough accumulator tensor.
            thetas (torch.Tensor): The angles (thetas) corresponding to dim 1 of the tensor.
            x_start (torch.Tensor): The starting index of the line.
            x_end (torch.Tensor): The ending index of the line.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The values from the accumulator along the line and the corresponding thetas.
        """
        H, W = accumulator.shape
        if x_start == x_end:
            return accumulator[:, x_start], thetas[x_start].repeat(H)

        x_coords = torch.linspace(x_start, x_end, steps=H)
        y_coords = torch.arange(H)
        sampled_values = accumulator[y_coords.long(), x_coords.round().long()]
        sampled_thetas = thetas[x_coords.round().long()]

        return sampled_values, sampled_thetas

    def line_intersection_from_hough(
        self, rho1: torch.Tensor, theta1: torch.Tensor, rho2: torch.Tensor, theta2: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Solves for the intersection point `(x, y)` of two lines represented in Hough space:

        .. math::

            x \sin(\theta_1) + y \cos(\theta_1) = \rho_1 \\
            x \sin(\theta_2) + y \cos(\theta_2) = \rho_2

        Args:
            rho1 (torch.Tensor): The radial distance of the first line.
            theta1 (torch.Tensor): The angle of the first line in radians.
            rho2 (torch.Tensor): The radial distance of the second line.
            theta2 (torch.Tensor): The angle of the second line in radians.

        Returns:
            torch.Tensor: The intersection point `(x, y)` of the two lines.
        """

        det = torch.cos(theta1) * torch.sin(theta2) - torch.cos(theta2) * torch.sin(theta1)
        if det.abs() < 1e-6:
            raise ValueError("Lines are parallel")
        x = (rho1 * torch.sin(theta2) - rho2 * torch.sin(theta1)) / det
        y = (rho2 * torch.cos(theta1) - rho1 * torch.cos(theta2)) / det
        return torch.stack((x, y))
