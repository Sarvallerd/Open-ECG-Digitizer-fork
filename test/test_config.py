from src.train import main
from src.config.default import get_cfg


def test_unet() -> None:
    cfg = get_cfg("./test/test_data/config/unet.yml")
    main(cfg)
