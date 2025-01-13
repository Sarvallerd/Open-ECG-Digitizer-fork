import torch
import torchvision


def filter_local_maxima(preds: torch.Tensor, distance: int) -> torch.Tensor:
    """
    Retain only local maxima within a specified distance in the input tensor.

    Args:
        preds (torch.Tensor): A 2D tensor representing the segmented probability map.
        distance (int): The radius within which to identify local maxima.

    Returns:
        torch.Tensor: A 2D tensor where only the local maxima are retained.
    """
    max_preds = torch.nn.functional.max_pool2d(
        preds.unsqueeze(0).unsqueeze(0), kernel_size=(distance, 1), stride=1, padding=(distance // 2, 0)
    ).squeeze()
    return preds * (preds == max_preds)


def find_closest_peak(nonzero_preds: torch.Tensor, index: int, num_peaks: int) -> int:
    """
    Args:
        nonzero_preds (torch.Tensor): A tensor of non-zero predictions.
        index (int): The current index being evaluated.
        num_peaks (int): The target number of peaks.

    Returns:
        int: The index of the closest matching peak.
    """
    for offset in range(1, len(nonzero_preds)):
        if (index - offset >= 0) and (nonzero_preds[index - offset] == num_peaks):
            return index - offset
        if (index + offset < len(nonzero_preds)) and (nonzero_preds[index + offset] == num_peaks):
            return index + offset
    return -1


def get_alignment(other_peaks: torch.Tensor, current_peaks: torch.Tensor) -> torch.Tensor:
    """
    Compute the optimal alignment between two sets of peaks using minimal differences.

    Args:
        diffs (torch.Tensor): A matrix of differences between two sets of peaks.

    Returns:
        torch.Tensor: Indices representing the alignment.
    """
    diffs = (other_peaks[:, None] - current_peaks[None, :]).abs()
    if diffs.shape[0] < diffs.shape[1]:
        indices: list[torch.Tensor] = []
        for row in diffs:
            if indices:
                row[: indices[-1] + 1] = float("inf")
            indices.append(torch.argmin(row))
        return torch.tensor(indices)
    else:
        return torch.argmin(diffs, dim=0)


class Snake(torch.nn.Module):
    def __init__(
        self,
        num_peaks: int = 6,
        autodetect_num_peaks: bool = False,
        autodetect_num_peaks_min: int = 3,
        autodetect_num_peaks_max: int = 6,
        min_distance: int = 45,
        left_percentile: float = 0.01,
        right_percentile: float = 0.99,
        top_percentile: float = 0.01,
        bottom_percentile: float = 0.99,
        horizontal_margin: int = 1,
        vertical_margin: int = 1,
        interpolate_missing: bool = True,
        matching_sim_threshold: float = 0.9,
        set_zero_threshold: float = 0.5,
    ):
        """
        Args:
            num_peaks (int): The desired number of peaks to identify (should be equal to number of ECG signals).
            min_distance (int): The minimum distance between signals during initialization.
            left_percentile (float): The percentile to crop the left side of the input tensor.
            right_percentile (float): The percentile to crop the right side of the input tensor.
            top_percentile (float): The percentile to crop the top side of the input tensor.
            bottom_percentile (float): The percentile to crop the bottom side of the input tensor.
            horizontal_margin (int): The number of pixels to extend the horizontal cropping.
            vertical_margin (int): The number of pixels to extend the vertical cropping.
            interpolate_missing (bool): Whether to interpolate missing chunks via pattern matching during initalization.
            matching_sim_threshold (float): The threshold for matching similarity.
            set_zero_threshold (float): The threshold for setting values in preds to zero, as a fraction of the maximum value in preds.
        """
        super(Snake, self).__init__()
        self.num_peaks = num_peaks
        self.autodetect_num_peaks = autodetect_num_peaks
        self.autodetect_num_peaks_min = autodetect_num_peaks_min
        self.autodetect_num_peaks_max = autodetect_num_peaks_max
        self.min_distance = min_distance
        self.left_percentile = left_percentile
        self.right_percentile = right_percentile
        self.top_percentile = top_percentile
        self.bottom_percentile = bottom_percentile
        self.horizontal_margin = horizontal_margin
        self.vertical_margin = vertical_margin
        self.matching_sim_threshold = matching_sim_threshold
        self.set_zero_threshold = set_zero_threshold
        self.interpolate_missing = interpolate_missing
        self._check_input()

    def fit(self, preds: torch.Tensor) -> None:
        """
        Args:
            preds (torch.Tensor): A 2D tensor of predictions, with values of magnitude within (0,1).
        """
        self.cropped_preds = self._crop_preds(preds)
        if self.autodetect_num_peaks:
            self._autodetect_num_peaks()
        self.snake = torch.nn.Parameter(self._initialize_snake()).to(preds.device)
        if self.interpolate_missing:
            self._interpolate_missing_chunks()

    def _check_input(self) -> None:
        assert 0 <= self.left_percentile <= 1, "left_percentile must be in range 0,1"
        assert 0 <= self.right_percentile <= 1, "right_percentile must be in range 0,1"
        assert 0 <= self.top_percentile <= 1, "top_percentile must be in range 0,1"
        assert 0 <= self.bottom_percentile <= 1, "bottom_percentile must be in range 0,1"
        assert 0 < self.num_peaks <= 12, "num_peaks must be in range 1,12"
        assert 0 < self.min_distance <= 200, "min_distance must be in range 1,200"
        assert 0 <= self.horizontal_margin, "horizontal_margin must be positive"
        assert 0 <= self.vertical_margin, "vertical_margin must be positive"
        assert 0 <= self.matching_sim_threshold <= 1, "matching_sim_threshold must be in range 0,1"
        assert 0 <= self.set_zero_threshold <= 1, "set_zero_threshold must be in range 0,1"

    def _crop_preds(self, preds: torch.Tensor) -> torch.Tensor:
        x_projection = preds.sum(0)
        y_projection = preds.sum(1)
        cumulative_sum_x = x_projection.cumsum(0) / x_projection.cumsum(0)[-1]
        cumulative_sum_y = y_projection.cumsum(0) / y_projection.cumsum(0)[-1]

        left_bound = int((cumulative_sum_x - self.left_percentile).abs().argmin().item())
        right_bound = int((cumulative_sum_x - self.right_percentile).abs().argmin().item())
        top_bound = int((cumulative_sum_y - self.top_percentile).abs().argmin().item())
        bottom_bound = int((cumulative_sum_y - self.bottom_percentile).abs().argmin().item())

        left_bound = max(0, left_bound - self.horizontal_margin)
        right_bound = min(preds.shape[1], right_bound + self.horizontal_margin)
        top_bound = max(0, top_bound - self.vertical_margin)
        bottom_bound = min(preds.shape[0], bottom_bound + self.vertical_margin)

        cropped_preds = preds[top_bound:bottom_bound, left_bound:right_bound]

        return cropped_preds

    def _find_contiguous_chunks(self) -> list[torch.Tensor]:
        nan_mask = torch.isnan(self.snake.data).any(0) & ~torch.isnan(self.snake.data).all(0)

        nan_indices = nan_mask.nonzero(as_tuple=True)[0]

        breaks = torch.diff(nan_indices) != 1
        chunk_splits = torch.where(breaks)[0] + 1  # Find split points

        start_idx = 0
        contiguous_chunks = []
        for split in chunk_splits:
            contiguous_chunks.append(nan_indices[start_idx:split])
            start_idx = split
        contiguous_chunks.append(nan_indices[start_idx:])  # Add last chunk

        return contiguous_chunks

    def _initialize_snake(self) -> torch.Tensor:
        """
        For each column in preds identify local maxima and, if needed,

        Args:
            preds (torch.Tensor): A 2D tensor of predictions.
            num_peaks (int): The desired number of peaks to identify.
            distance (int): The minimum distance between peaks.
            set_zero_threshold (float): The threshold for setting values to zero.

        Returns:
            torch.Tensor: A 2D tensor containing the aligned peaks.
        """
        preds = filter_local_maxima(self.cropped_preds, self.min_distance)
        preds[preds < preds.max() * self.set_zero_threshold] = 0

        nonzero_preds = (preds > 0).sum(0)
        peaks = preds.argsort(dim=0, stable=True, descending=True).float()

        matching_num_peaks = (nonzero_preds == self.num_peaks).nonzero(as_tuple=True)[0]
        if len(matching_num_peaks) >= 2:
            first_index = matching_num_peaks[0]
            last_index = matching_num_peaks[-1]
        else:  # sets nan values everywhere
            first_index = torch.tensor(-1)
            last_index = torch.tensor(0)

        for i in range(peaks.shape[1]):
            if nonzero_preds[i] != self.num_peaks:
                if nonzero_preds[i] >= self.num_peaks * 2:
                    peaks[:, i] = torch.nan
                    continue

                index = find_closest_peak(nonzero_preds, i, self.num_peaks)
                if index == -1:
                    peaks[: self.num_peaks, i] = torch.nan
                    continue

                other_peaks = peaks[: self.num_peaks, index].sort(dim=0).values.clone()
                current_peaks = peaks[: nonzero_preds[i], i].sort(dim=0).values.clone()

                indices = get_alignment(other_peaks, current_peaks)
                if len(indices) == self.num_peaks:
                    peaks[: self.num_peaks, i] = current_peaks[indices]
                    nonzero_preds[i] = self.num_peaks
                else:
                    peaks[: self.num_peaks, i] = torch.nan
                    peaks[indices, i] = current_peaks[: len(indices)]
            else:
                peaks[: self.num_peaks, i] = peaks[: self.num_peaks, i].sort(dim=0).values

        peaks[:, :first_index] = torch.nan
        peaks[:, last_index + 1 :] = torch.nan

        return peaks[: self.num_peaks]

    def _interpolate_missing_chunks(self) -> None:
        self._postprocess_initialization()
        chunks = self._find_contiguous_chunks()
        snake_interpolation = torch.full_like(self.snake.data, torch.nan)
        all_invalid_indices = torch.isnan(self.snake.data).any(0)

        for chunk in chunks:
            valid_channels = ~torch.isnan(self.snake.data[:, chunk]).any(1)

            kernel_to_match = self.snake.data[valid_channels][:, chunk].clone()

            kernel_to_match -= kernel_to_match.mean()
            search_buffer = self.snake.data[valid_channels].clone()
            search_buffer[:, all_invalid_indices] = torch.nan

            best_match_index = 0
            best_match_sim = float("-inf")
            for i in range(search_buffer.shape[1] - kernel_to_match.shape[1]):
                if torch.isnan(search_buffer[:, i : i + kernel_to_match.shape[1]]).any():
                    continue
                search_window = search_buffer[:, i : i + kernel_to_match.shape[1]].clone()
                search_window -= search_window.mean()
                sim = torch.nn.functional.cosine_similarity(kernel_to_match, search_window).mean().item()

                if sim > best_match_sim:
                    best_match_sim = sim
                    best_match_index = i

            if best_match_sim > self.matching_sim_threshold:
                snake_interpolation[:, chunk] = self.snake.data[
                    :, best_match_index : best_match_index + kernel_to_match.shape[1]
                ]
                if chunk[0] - 1 >= 0 and chunk[-1] + 1 < self.snake.data.shape[1]:
                    diff1 = snake_interpolation[:, chunk[0]] - self.snake.data[:, chunk[0] - 1]
                    diff2 = snake_interpolation[:, chunk[-1]] - self.snake.data[:, chunk[-1] + 1]
                    diff = torch.stack([diff1, diff2]).nanmean(0)
                    snake_interpolation[:, chunk] -= diff[:, None]

        self.snake.data[torch.isnan(self.snake.data)] = snake_interpolation[torch.isnan(self.snake.data)]

    def _postprocess_initialization(self) -> None:
        """Sets values in the snake.data tensor to nan if they are more than halfway towards the next channel median."""
        channel_medians = torch.nanmedian(self.snake.data, dim=1).values
        diff = channel_medians.diff().abs().mean().item()
        for c in range(self.snake.data.shape[0]):
            self.snake.data[c][(self.snake.data[c] - channel_medians[c]).abs() > diff / 2] = torch.nan

    def _autodetect_num_peaks(self) -> None:
        """Projects the cropped_preds tensor by summation, smooths the projection and sets number of peaks based on number of local maxima."""
        preds_copy = self.cropped_preds.clone()
        preds_copy[preds_copy < 2 * preds_copy.max() / 3] = 0
        projected_preds = preds_copy.pow(2).sum(1, keepdim=True)
        first, last = projected_preds.nonzero(as_tuple=True)[0][[0, -1]]
        projected_preds = projected_preds[first : last + 1]
        base_kernel_size = projected_preds.shape[0] // 3
        if not base_kernel_size % 2:
            base_kernel_size += 1
        projected_preds = torch.nn.functional.pad(
            projected_preds, (0, 0, base_kernel_size // 4, base_kernel_size // 4), value=projected_preds.min().item()
        )
        blurred_preds = torchvision.transforms.functional.gaussian_blur(
            projected_preds.unsqueeze(0).unsqueeze(0),
            kernel_size=(1, base_kernel_size),
            sigma=(1, base_kernel_size // 8 + 1),
        ).squeeze()

        # count the number of local maxima
        max_pooled = torch.nn.functional.max_pool1d(
            blurred_preds.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1
        ).squeeze()
        max_indices = (blurred_preds == max_pooled).nonzero(as_tuple=True)[0]
        num_peaks = len(max_indices)

        self.num_peaks = min(max(num_peaks, self.autodetect_num_peaks_min), self.autodetect_num_peaks_max)

    def forward(self) -> torch.Tensor:
        self.snake.data = torch.sort(self.snake.data, dim=0)[0]
        return self.snake
