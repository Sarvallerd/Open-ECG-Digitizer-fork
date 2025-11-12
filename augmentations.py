import json
from pathlib import Path
import math
import random
import os
import cv2 
import multiprocessing
import albumentations as A
from augraphy import (
    LCDScreenPattern,
    LightingGradient,
    ShadowCast,
    BadPhotoCopy
)
from tqdm import tqdm





# Comment these 3 lines pls and write correct names:
# -------------------------------------------
BASE = Path('/home/jovyan/shares/SR003.nfs2/kandi_image/T2I_Whatislove/sub/ecg_lib/ecg-image-kit/codes/ecg-image-generator')
no_aug_path = Path('/home/jovyan/shares/SR003.nfs2/kandi_image/T2I_Whatislove/sub/no_aug_data')
aug_path = Path('/home/jovyan/shares/SR003.nfs2/kandi_image/T2I_Whatislove/sub/aug_data')
# -------------------------------------------

lcdscreenpattern = LCDScreenPattern(pattern_type="Vertical",
                                    pattern_value_range = (8,16),
                                    pattern_skip_distance_range = (3,5),
                                    p=0.5
                                    )
lighting_gradient_gaussian = LightingGradient(light_position=None,
                                              direction=90,
                                              max_brightness=255,
                                              min_brightness=0,
                                              mode="gaussian",
                                              transparency=0.5,
                                              p=0.5

                                              )

shadowcast = ShadowCast(shadow_side = "bottom",
                        shadow_vertices_range = (2, 3),
                        shadow_width_range=(0.5, 0.8),
                        shadow_height_range=(0.5, 0.8),
                        shadow_color = (0, 0, 0),
                        shadow_opacity_range=(0.4,0.6),
                        shadow_iterations_range = (1,2),
                        shadow_blur_kernel_range = (200, 301),
                        p = 0.5
                        )


def goida_augment(image_path, mask_path, scale, backgrounds_path=BASE/'backgrounds',
                    output_dir=BASE/'augdata', num_augmentations=2, start_index=0):
    os.makedirs(output_dir, exist_ok=True)
    original_name = image_path.name.split('_')[0]
    if list(output_dir.glob(f'{original_name}*')):
        return
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    cv2.imwrite(output_dir/f'{original_name}_{start_index:05d}.png', image)
    cv2.imwrite(output_dir/f'{original_name}_{start_index:05d}_mask.png', mask)
    image = lcdscreenpattern(image)
    image = lighting_gradient_gaussian(image)
    image = shadowcast(image)
    backgrounds_names = os.listdir(backgrounds_path)
    random_background = random.choice(backgrounds_names)
    background = cv2.resize(cv2.imread(backgrounds_path/random_background), (2200, 1700))
    reduced_width, reduced_height = int(2200 * scale), int(1700 * scale)
    background[1700 // 2 - reduced_height // 2 : 1700 // 2 + reduced_height // 2, 2200 // 2 - reduced_width // 2 : 2200 // 2 + reduced_width // 2] = 0
    affine_transform = A.Compose([
        A.Affine(
            translate_percent=0,
            scale=scale,
            rotate=0,
            shear=0,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=1
        ),
    ], additional_targets={
        'mask': 'mask',
    })
    after_affine = affine_transform(
                image=image,
                mask=mask,
            )
    image = after_affine['image'] + background
    mask = after_affine['mask']
    rotation = A.Compose([
        A.Rotate(limit=20, p=1),
        A.GridDistortion(
            num_steps=2, 
            distort_limit=0.3, 
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=1
        ),
    ], additional_targets={
        'mask': 'mask',
    })
    after_rotation = rotation(
        image=image,
        mask=mask,
    )
    image = after_rotation['image']
    mask = after_rotation['mask']
    transform = A.Compose([
        A.ColorJitter(
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=(-0.5, 0.5),
            p=0.5
        ),
        A.ToGray(p=0.5),
        
        A.RandomGamma(gamma_limit=(80, 120), p=1),
    ])
    for i in range(start_index + 1, start_index + num_augmentations + 1):
        augmented = transform(
            image=image,
            mask=mask,
        )
        cv2.imwrite(output_dir/f'{original_name}_{i:05d}.png', augmented['image'])
        cv2.imwrite(output_dir/f'{original_name}_{i:05d}_mask.png', augmented['mask'])
    return start_index + num_augmentations

def multi_goida_augment(args):
    image_path, mask_path, folder, start_index, num_augmentations = args
    try:
        goida_augment(
        image_path=image_path,
        mask_path=mask_path,
        scale=0.75,
        output_dir=folder,
        num_augmentations=num_augmentations,
        start_index=start_index
        )
    except:
        return

def create_augmented_dataset(no_aug_data_path: Path,
                             aug_data_path: Path,
                             train_test_ratio: float = 0.7,
                             n_processes: int = 40,
                             num_augmentations : int = 10
) -> None:
    work = []
    aug_data_path.mkdir(parents=True, exist_ok=True)
    (aug_data_path/'train').mkdir(parents=True, exist_ok=True)
    (aug_data_path/'val').mkdir(parents=True, exist_ok=True)
    
    n_total = len(list(no_aug_data_path.glob('*.hea')))
    processed_files = 0
    
    for filename in no_aug_data_path.glob('*.hea'):
        base_name = filename.stem
        image_file = no_aug_data_path/f'{base_name}-0.png'
        mask_file = no_aug_data_path/f'{base_name}-0_mask.png'
        
        if image_file.is_file() and mask_file.is_file():
            folder = aug_data_path/('train' if processed_files < int(train_test_ratio * n_total) else 'val')
            start_index = processed_files
            
            work.append(
                (
                    image_file,
                    mask_file,
                    folder,
                    start_index,
                    num_augmentations
                )
            )
            processed_files += 1
    
    print(f"Всего файлов для обработки: {len(work)}")
    multiprocessing.Pool(n_processes).map(multi_goida_augment, work)

if __name__ == '__main__':
    create_augmented_dataset(
        Path('/home/jovyan/shares/SR003.nfs2/kandi_image/T2I_Whatislove/sub/no_aug_data'),
        Path('/home/jovyan/shares/SR003.nfs2/kandi_image/T2I_Whatislove/sub/aug_data'),
        train_test_ratio=0.7,
        n_processes=15,
        num_augmentations=5
    )