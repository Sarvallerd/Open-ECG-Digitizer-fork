# Electrocardiogram-Digitization

![Tests](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/actions/workflows/test.yml/badge.svg?branch=main) ![](https://img.shields.io/badge/%20style-google-3666d6.svg)

## Run a training session
1. Run `python3 -m src.train` in the terminal from the repository root folder.
2. The results can be visualized with tensorboard.
    1. If you are running on the command on a VM; `ssh -L 16006:127.0.0.1:6006 <username>@<vm ip>` will forward the tensorboard traffic to your machine.
    2. Launch tensorboard `tensorboard --logdir <logdir of run>`
    3. Tensorboard is available on [localhost:16006](localhost:16006).