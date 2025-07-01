from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


class LeadIdentifier:
    LEAD_CHANNEL_ORDER: list[str] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    def __init__(
        self,
        *,
        layouts: dict[str, Any],
        unet: torch.nn.Module,
        device: torch.device,
        debug: bool = False,
    ) -> None:
        """
        Args:
            layouts: dict mapping layout names → layout definitions.
            unet: a loaded and weight-initialized UNet model (in eval mode).
            device: device where inference will run.
            debug: whether to draw scatter/plots.
        """
        self.layouts = layouts
        self.unet = unet.to(device).eval()
        self.device = device
        self.debug = debug

    def _merge_nonoverlapping_lines(self, lines: torch.Tensor) -> torch.Tensor:
        """Merges adjacent non-overlapping lines.

        Args:
            lines: Input tensor of lines.

        Returns:
            Tensor with merged lines.
        """
        if lines.shape[0] > 1:
            means: torch.Tensor = torch.nanmean(lines, dim=1)
            sorted_indices: torch.Tensor = torch.argsort(means)
            lines = lines[sorted_indices]
        changed: bool = True
        while changed and lines.shape[0] > 1:
            changed = False
            new_lines: list[torch.Tensor] = []
            i = 0
            while i < lines.shape[0]:
                if i < lines.shape[0] - 1:
                    row1 = lines[i]
                    row2 = lines[i + 1]
                    overlap = ~(torch.isnan(row1) | torch.isnan(row2))
                    if not torch.any(overlap):
                        # Merge: prefer values from row1 where not nan, else row2
                        merged = torch.where(torch.isnan(row1), row2, row1)
                        new_lines.append(merged)
                        i += 2
                        changed = True
                        continue
                new_lines.append(lines[i])
                i += 1
            lines = torch.stack(new_lines)
        return lines

    def _nan_cossim(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Calculates cosine similarity, ignoring NaNs.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            Cosine similarity as float.
        """
        x_values: torch.Tensor = x[~torch.isnan(x)]
        y_values: torch.Tensor = y[~torch.isnan(y)]
        if x_values.numel() <= 1 or y.numel() <= 1:
            return -1.0
        x_mean: torch.Tensor = x_values.mean()
        y_mean: torch.Tensor = y_values.mean()
        x_norm: torch.Tensor = x - x_mean
        y_norm: torch.Tensor = y - y_mean
        both_exist_mask: torch.Tensor = ~(torch.isnan(x_norm) | torch.isnan(y_norm))
        if not torch.any(both_exist_mask):
            return -1.0
        x_norm = x_norm[both_exist_mask]
        y_norm = y_norm[both_exist_mask]
        x_norm = x_norm / torch.linalg.norm(x_norm, keepdim=True)
        y_norm = y_norm / torch.linalg.norm(y_norm, keepdim=True)
        return float(torch.dot(x_norm, y_norm).item())

    def _canonicalize_lines(self, lines: torch.Tensor, match: dict[str, Any]) -> torch.Tensor:
        """Reorders and flips lines to a canonical 12-lead format.

        Args:
            lines: Tensor of lines.
            match: Layout match info.

        Returns:
            Canonicalized tensor.
        """
        canonical_order: list[str] = self.LEAD_CHANNEL_ORDER
        num_leads: int = len(canonical_order)
        width: int = lines.shape[1]
        canonical: torch.Tensor = torch.full((num_leads, width), float("nan"), dtype=lines.dtype, device=lines.device)

        layout_name: Optional[str] = match.get("layout")
        flip: bool = match.get("flip", False)
        if layout_name is None or layout_name not in self.layouts:
            return canonical
        layout_def: dict[str, Any] = self.layouts[layout_name]
        leads_def: Any = layout_def["leads"]
        rhythm_leads: list[str] = layout_def.get("rhythm_leads", [])
        cols: int = layout_def["layout"]["cols"]

        if flip:
            lines = torch.flip(lines, dims=[0, 1])
            lines_max_val = lines[~torch.isnan(lines)].max()
            lines = lines_max_val - lines

        lead_names: list[str] = []
        for row in leads_def:
            if isinstance(row, list):
                lead_names.extend(row)
            else:
                lead_names.append(row)
        lead_names += rhythm_leads

        used_indices: set[tuple[int, int, int]] = set()

        # Use the layout that we have matched, to fill a "canonical" tensor, in which the leads are always ordered as self.LEAD_CHANNEL_ORDER.
        chunk_width: int = width // cols
        for row_idx, layout_row in enumerate(leads_def):
            if not isinstance(layout_row, list):
                layout_row = [layout_row]
            for col_idx, layout_lead in enumerate(layout_row):
                start: int = col_idx * chunk_width
                end: int = (col_idx + 1) * chunk_width if col_idx < cols - 1 else width
                if row_idx >= lines.shape[0]:
                    continue  # Incomplete
                chunk: torch.Tensor = lines[row_idx, start:end]
                sign: int = 1
                lead_name: str = layout_lead
                if isinstance(lead_name, str) and lead_name.startswith("-"):
                    sign = -1
                    lead_name = lead_name[1:]
                if lead_name in canonical_order:
                    canon_idx: int = canonical_order.index(lead_name)
                    canonical[canon_idx, start:end] = sign * chunk
                    used_indices.add((canon_idx, start, end))

        # If there are any rythm, leads, we try to match them with the canonical leads, through cosine similarity.
        num_rhythm_leads: int = len(rhythm_leads)
        if num_rhythm_leads > 0:
            rhythm_corrs: NDArray[np.float32] = np.full((num_rhythm_leads, num_leads), -1, dtype=np.float32)
            for i in range(num_rhythm_leads):
                rhythm_vec: torch.Tensor = lines[-num_rhythm_leads + i, :]
                for j in range(num_leads):
                    corr: float = self._nan_cossim(rhythm_vec, canonical[j, :])
                    rhythm_corrs[i, j] = corr

            # Leads II, V1 and V5 are most commonly used for rhythm, so we inflate their cosine similarity.
            # Other matches are still possible.
            if num_rhythm_leads == 1:
                rhythm_corrs[:, 1] = self._inflate_cossim(rhythm_corrs[:, 1])
            elif num_rhythm_leads == 2:
                rhythm_corrs[:, 1] = self._inflate_cossim(rhythm_corrs[:, 1])
                rhythm_corrs[:, 6] = self._inflate_cossim(rhythm_corrs[:, 6])
            elif num_rhythm_leads == 3:
                rhythm_corrs[:, 1] = self._inflate_cossim(rhythm_corrs[:, 1])
                rhythm_corrs[:, 6] = self._inflate_cossim(rhythm_corrs[:, 6])
                rhythm_corrs[:, 10] = self._inflate_cossim(rhythm_corrs[:, 10])

            row_idx, col_idx = linear_sum_assignment(-rhythm_corrs)
            for i_r, i_c in zip(row_idx, col_idx):
                corr_val: float = rhythm_corrs[i_r, i_c]
                print(f"Rhythm {i_r} → Canonical {canonical_order[i_c]} (corr={corr_val:.2f})")
                canonical[i_c, :] = lines[-num_rhythm_leads + i_r, :]

        return canonical

    def _inflate_cossim(
        self, lead: Union[float, NDArray[Any], torch.Tensor], factor: float = 0.75
    ) -> Union[float, NDArray[Any], torch.Tensor]:
        """Inflates cosine similarity by a given factor.

        Args:
            lead: Cosine similarity value(s).
            factor: Scaling factor.

        Returns:
            Inflated value(s).
        """
        return 1 - factor + factor * lead

    def _generate_grid_positions(self, layout_def: dict[str, Any]) -> dict[str, NDArray[np.float64]]:
        """Generates normalized grid positions for each lead.

        Args:
            layout_def: Layout definition dict.

        Returns:
            Mapping of lead name to grid position array.
        """
        rows: int = layout_def["layout"]["rows"]
        cols: int = layout_def["layout"]["cols"]
        leads: Any = layout_def["leads"]

        def norm_y(i: int) -> float:
            return i / (rows - 1) if rows > 1 else 0.5

        def norm_x(j: int) -> float:
            return j / (cols - 1) if cols > 1 else 0.5

        pos: dict[str, NDArray[np.float64]] = {}
        lead_str: str
        if isinstance(leads[0], list):
            for y_idx, row in enumerate(leads):
                for x_idx, lead in enumerate(row):
                    lead_str = lead.strip("-")
                    pos[lead_str] = np.array([norm_x(x_idx), norm_y(y_idx)], dtype=np.float64)
        elif len(leads) == rows * cols:
            for idx, lead in enumerate(leads):
                y_idx, x_idx = divmod(idx, cols)
                lead_str = lead.strip("-")
                pos[lead_str] = np.array([norm_x(x_idx), norm_y(y_idx)], dtype=np.float64)
        else:
            for y_idx, lead in enumerate(leads):
                lead_str = lead.strip("-")
                pos[lead_str] = np.array([0.5, norm_y(y_idx)], dtype=np.float64)
        return pos

    def _extract_lead_points(
        self, probs_tensor: torch.Tensor, lead_names: Optional[list[str]] = None
    ) -> list[tuple[str, float, float]]:
        """Extracts center-of-mass points for each lead.

        Args:
            probs_tensor: Probability tensor from UNet.
            lead_names: List of lead names.

        Returns:
            List of (lead name, x, y) tuples.
        """
        if lead_names is None:
            lead_names = self.LEAD_CHANNEL_ORDER
        _, C, H, W = probs_tensor.shape
        arr: NDArray[Any] = probs_tensor[0].cpu().numpy()
        pts: list[tuple[str, float, float]] = []
        for i, name in enumerate(lead_names):
            channel: NDArray[Any] = arr[i]
            if np.sum(channel) == 0:
                continue
            x_fmap: NDArray[Any] = np.arange(W).reshape(1, W).repeat(H, axis=0)
            y_fmap: NDArray[Any] = np.arange(H).reshape(H, 1).repeat(W, axis=1)
            x_com: float = float(np.sum(x_fmap * channel) / np.sum(channel))
            y_com: float = float(np.sum(y_fmap * channel) / np.sum(channel))
            pts.append((name, x_com, y_com))
        return pts

    def _match_layout(self, detected_pts: list[tuple[str, float, float]], rows_in_layout: int) -> dict[str, Any]:
        """Finds the best matching layout for detected lead points.

        Args:
            detected_pts: List of detected (lead, x, y) tuples.
            rows_in_layout: Number of rows in the layout.

        Returns:
            Dict with match information.
        """
        names: tuple[str, ...]
        xs: tuple[float, ...]
        ys: tuple[float, ...]
        names, xs, ys = zip(*detected_pts)
        pts: NDArray[np.float64] = np.stack([xs, ys], axis=1)
        n: int = pts.shape[0]
        best: dict[str, Any] = {"cost": np.inf}

        # Loop through each predefined possible layout, and calculate how fitting each of them is.
        for layout_name, desc in self.layouts.items():
            total_rows: int = desc["total_rows"]
            rows_difference: int = abs(total_rows - rows_in_layout)
            if rows_difference > 1:
                continue

            pos_map: dict[str, NDArray[np.float64]] = self._generate_grid_positions(desc)
            grid_leads: list[str] = list(pos_map.keys())
            grid_pts: NDArray[np.float64] = np.stack([pos_map[lead] for lead in grid_leads])
            scaling_factor: float = max(len(grid_leads), n) / min(len(grid_leads), n) * (1 + rows_difference * 3)

            for flip in (False, True):
                P: NDArray[np.float64] = pts.copy()
                if flip:
                    P = -P

                # This first part aims at scaling the coordinates to find the best match, as
                # we do not a priori know where the leads are in the image.
                Pm: list[NDArray[np.float64]] = []
                Gm: list[NDArray[np.float64]] = []
                idxs: list[tuple[int, int]] = []
                missing: int = 0
                for i, lead in enumerate(names):
                    if lead in pos_map:
                        j = grid_leads.index(lead)
                        Pm.append(P[i])
                        Gm.append(grid_pts[j])
                        idxs.append((i, j))
                    else:
                        missing += 1
                Pm_arr: NDArray[np.float64] = np.array(Pm)
                Gm_arr: NDArray[np.float64] = np.array(Gm)
                if Pm_arr.shape[0] < 2:
                    continue

                mu_P: NDArray[np.float64] = Pm_arr.mean(axis=0)
                mu_G: NDArray[np.float64] = Gm_arr.mean(axis=0)
                Pc: NDArray[np.float64] = Pm_arr - mu_P
                Gc: NDArray[np.float64] = Gm_arr - mu_G
                num: NDArray[np.float64] = np.sum(Pc * Gc, axis=0)
                den: NDArray[np.float64] = np.sum(Pc**2, axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    s: NDArray[np.float64] = num / den
                s = np.where(np.isfinite(s), s, 0.0)
                if np.any(s < 0):
                    continue
                s[s < 1e-4] = 1e-4
                t: NDArray[np.float64] = mu_G - s * mu_P
                P_scaled: NDArray[np.float64] = P * s + t

                # After scaling the coordinates, we calculate the distances between the detected leads and the grid leads.
                res: list[float] = []
                for i, j in idxs:
                    res.append(float(np.linalg.norm(P_scaled[i] - grid_pts[j])))
                PENALTY: float = 0.5
                res.extend([PENALTY] * missing)
                avg_res: float = float(np.mean(res)) * scaling_factor

                if avg_res < best["cost"]:
                    best = {"layout": layout_name, "flip": flip, "cost": avg_res, "leads": grid_leads}
                if self.debug:
                    plt.scatter(grid_pts[:, 0], grid_pts[:, 1], c="blue", label="Layout", s=60)
                    plt.scatter(P_scaled[:, 0], P_scaled[:, 1], c="red", label="Detected", s=60)
                    for ii, jj in idxs:
                        plt.plot(
                            [P_scaled[ii, 0], grid_pts[jj, 0]], [P_scaled[ii, 1], grid_pts[jj, 1]], c="gray", alpha=0.5
                        )
                        plt.text(P_scaled[ii, 0], P_scaled[ii, 1], names[ii], ha="right", va="bottom", fontsize=9)
                        plt.text(grid_pts[jj, 0], grid_pts[jj, 1], grid_leads[jj], ha="left", va="top", fontsize=9)
                    plt.gca().invert_yaxis()
                    plt.title(f"{layout_name} (flip={flip}) - cost={avg_res:.4f}")
                    plt.legend()
                    plt.show()
        return best

    def normalize(self, lines: torch.Tensor, avg_pixel_per_mm: float, mv_per_mm: float) -> torch.Tensor:
        """Normalizes lines from pixel to mV scale.

        Args:
            lines: Input tensor of lines.
            avg_pixel_per_mm: Average pixel per mm.
            mv_per_mm: mV per mm.

        Returns:
            Normalized tensor.
        """
        lines = lines - lines.nanmean(dim=1, keepdim=True)
        lines = lines * (mv_per_mm / avg_pixel_per_mm)  # convert from pixels to mV
        return lines

    def __call__(
        self,
        lines: torch.Tensor,
        feature_map: torch.Tensor,
        avg_pixel_per_mm: float,
        threshold: float = 0.9,
        mv_per_mm: float = 0.1,
    ) -> dict[str, Any]:
        """Detects layout, canonicalizes lines, and returns results.

        Args:
            lines: Input lines tensor.
            feature_map: Image tensor for UNet.
            avg_pixel_per_mm: Average pixel per mm.
            threshold: UNet probability threshold.
            mv_per_mm: mV per mm.

        Returns:
            Dictionary with layout match info and canonical lines.
        """
        lines = self._merge_nonoverlapping_lines(lines)
        rows_in_layout: int = lines.shape[0]
        self.unet.eval()
        with torch.no_grad():
            logits: torch.Tensor = self.unet(feature_map.to(self.device))  # [1,13,H,W]
            probs: torch.Tensor = torch.softmax(logits, dim=1)[:, :12]  # [1,12,H,W]
            probs[:, 0] = (
                0  # Ignore the position of the "I" lead as it is particularly prone to false positives ("I" looks like a vertical line).
            )
        probs[probs < threshold] = 0
        detected: list[tuple[str, float, float]] = self._extract_lead_points(probs, self.LEAD_CHANNEL_ORDER)
        if len(detected) <= 2:
            match: dict[str, Any] = {"cost": float("inf")}
            canonical_lines: Optional[torch.Tensor] = None
        else:
            match = self._match_layout(detected, rows_in_layout)
            canonical_lines = self._canonicalize_lines(lines, match)
            canonical_lines = self.normalize(canonical_lines, avg_pixel_per_mm, mv_per_mm)

        return {
            "rows_in_layout": rows_in_layout,
            "n_detected": len(detected),
            **match,
            "canonical_lines": canonical_lines,
        }
