from src.train import main
from src.config.default import get_cfg


def test_unet() -> None:
    cfg = get_cfg("./test/test_data/config/unet.yml")
    main(cfg)


def test_unet_compilation() -> None:
    cfg = get_cfg("./test/test_data/config/unet_compilation.yml")
    main(cfg)
