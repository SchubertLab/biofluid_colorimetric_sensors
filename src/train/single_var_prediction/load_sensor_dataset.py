import torch
from torchvision import datasets, transforms
import src.data.data_loader as dataloader


def load_dataset(params):
    # Create Data Transforms
    composed_transforms = transforms.Compose([
        transforms.RandomRotation(
            degrees=(
                params['augmentation_specs']['image_rotation'][0],
                params['augmentation_specs']['image_rotation'][1]
            )
        ),
        transforms.RandomPerspective(
            distortion_scale=params['augmentation_specs']['distortion_scale'],
            p=params['augmentation_specs']['distortion_p'],
            interpolation=transforms.InterpolationMode.NEAREST,
        )
    ])

    # Load Dataset
    train_set = dataloader.DatasetSyntheticSensors(
        root_path=params['root_path'],
        paths_list_ref_color_dict=params['paths_list_ref_color_dict'],
        path_image_skeletons=params['path_image_skeletons'],
        label_dictionary=params['label_dictionary'],
        additional_specs=params['additional_specs'],
        n_synthetic_images=params['n_synthetic_images_train'],
        target_image_shape=params['target_image_shape'],
        n_sensing_areas=14,
        image_skeletons_format='.png',
        allow_incomplete=params['allow_incomplete'],
        incomplete_every_n=params['incomplete_every_n'],
        transform=composed_transforms,
    )

    test_set = dataloader.DatasetSyntheticSensors(
        root_path=params['root_path'],
        paths_list_ref_color_dict=params['paths_list_ref_color_dict'],
        path_image_skeletons=params['path_image_skeletons'],
        label_dictionary=params['label_dictionary'],
        additional_specs=params['additional_specs'],
        n_synthetic_images=params['n_synthetic_images_val'],
        target_image_shape=params['target_image_shape'],
        n_sensing_areas=14,
        image_skeletons_format='.png',
        allow_incomplete=params['allow_incomplete'],
        incomplete_every_n=params['incomplete_every_n'],
        transform=composed_transforms,
    )

    # Dataloader
    cuda_kwargs = {
        'batch_size': params['batch_size']
    }

    train_dataloader = dataloader.DataLoader(
        train_set,
        **cuda_kwargs,
    )
    test_dataloader = dataloader.DataLoader(
        test_set,
        **cuda_kwargs,
    )

    return train_dataloader, test_dataloader