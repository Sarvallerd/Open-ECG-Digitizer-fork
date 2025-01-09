from typing import Any, List, Tuple, Union
from torchvision.io import decode_image
import numpy as np
import os
import torch


class ECGScanDataset(torch.utils.data.Dataset[Any]):
    def __init__(
        self, ecg_scan_path: str, ecg_mask_path: str, transform: Any = None, load_n: Union[None, int] = None
    ) -> None:
        self.ecg_scan_path = ecg_scan_path
        self.ecg_mask_path = ecg_mask_path
        self.transform = transform

        # Prepend current software directory if path is not absolute.
        if not os.path.isabs(self.ecg_scan_path):
            self.ecg_scan_path = os.path.join(os.path.dirname(__file__), self.ecg_scan_path)
        if not os.path.isabs(self.ecg_mask_path):
            self.ecg_mask_path = os.path.join(os.path.dirname(__file__), self.ecg_mask_path)

        self.ecg_scan_files, self.ecg_mask_files = self._find_ecg_files()
        if load_n:
            self.ecg_scan_files = self.ecg_scan_files[:load_n]
            self.ecg_mask_files = self.ecg_mask_files[:load_n]

    def _find_ecg_files(self) -> Tuple[List[str], List[str]]:
        for path in [self.ecg_scan_path, self.ecg_mask_path]:
            if not os.path.exists(path):
                raise ValueError(f"Path {path} does not exist.")

        def is_no_underscore_npy_file(file: str) -> bool:
            return not file.startswith("_") and file.endswith(".npy")

        ecg_mask_filenames = list(
            map(lambda x: x.split(".")[0], filter(is_no_underscore_npy_file, os.listdir(self.ecg_mask_path)))
        )

        ecg_scan_files = list(map(lambda x: os.path.join(self.ecg_scan_path, f"{x}.png"), ecg_mask_filenames))
        ecg_mask_files = list(map(lambda x: os.path.join(self.ecg_mask_path, f"{x}.npy"), ecg_mask_filenames))
        return ecg_scan_files, ecg_mask_files

    def _load_scan(self, file: str) -> torch.Tensor:
        scan: torch.Tensor = decode_image(os.path.join(self.ecg_scan_path, file), mode="RGB").float() / 255.0
        return scan

    def _load_mask(self, file: str) -> torch.Tensor:
        loaded_mask = torch.tensor(np.load(file)).float()
        if loaded_mask.shape[0] != 3:
            loaded_mask = self._mask_to_one_hot(loaded_mask)
        return loaded_mask

    def _mask_to_one_hot(self, mask: torch.Tensor) -> torch.Tensor:
        num_classes = int(torch.max(mask)) + 1
        one_hot_mask: torch.Tensor = torch.nn.functional.one_hot(mask.long(), num_classes).permute(2, 0, 1).float()
        return one_hot_mask

    def __len__(self) -> int:
        return len(self.ecg_scan_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scan = self._load_scan(self.ecg_scan_files[idx])
        mask = self._load_mask(self.ecg_mask_files[idx])

        if self.transform:
            scan, mask = self.transform(scan, mask)
        return scan, mask
