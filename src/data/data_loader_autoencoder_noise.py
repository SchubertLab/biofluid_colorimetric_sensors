import os
import torch
import pickle
import random
import numpy as np
import pandas as pd
import skimage as ski
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import src.utils.label_areas_utils as label_utils
import src.data.light_kernels as light_kernels


class DatasetSyntheticSensors(Dataset):
    """
        Dataset class for SyntheticSensors
        input:
        output: sensor_distributions  skeletons_dict sensor_mask_per_variable and label_dictionary
    """

    def __init__(self, root_path, paths_list_ref_color_dict, path_image_skeletons, label_dictionary,
                 additional_specs, n_synthetic_images, target_image_shape,
                 n_sensing_areas=14, image_skeletons_format='.png',
                 allow_incomplete=True, incomplete_every_n=10, offset=0.1,
                 transform=None, illumination_noise=False, noise_level=0.5,
                 masked_ground_truth=False, one_mask_input=False,
                 ):

        # Init initializes (1) the self.sensor_distributions, (2) self.skeletons_dict to generate the synthetic images
        # and (3) the self.label_dictionary and additional variables.

        # 1. load the pixel distributions per variable empty_distributions_sampled
        sensor_distributions = {}
        for lsk in additional_specs['labels_to_sample_all'].keys():
            sensor_distributions[lsk] = {}
        list_repeated_areas = additional_specs['list_repeated_areas']
        for key in sensor_distributions:
            pixel_path = os.path.join(root_path, paths_list_ref_color_dict[key])
            with open(pixel_path, 'rb') as file:
                dictionary_values_per_image = pickle.load(file)
            sensor_distributions[key]['pixel_values'] = dictionary_values_per_image
            sensor_distributions[key]['repeated_areas'] = list_repeated_areas[key]

        self.sensor_distributions = sensor_distributions

        # 2. load the skeletons of the sensors from the path
        complete_path_image_skeletons = os.path.join(root_path, path_image_skeletons)
        file_names = sorted(
            [fname for fname in os.listdir(complete_path_image_skeletons) if fname.endswith(image_skeletons_format)]
        )
        skeletons_dict = {}
        for file in file_names:
            skeletons_dict[file] = {
                'rgb_image': [],
                'grayscale_image': [],
                'sensing_area_mask': [],
                'df_circles': [],
                'labeled_array': [],
                'masks_per_variable': [],
            }
            temp_file_path = os.path.join(complete_path_image_skeletons, file)
            temp_image = ski.io.imread(temp_file_path)
            if temp_image.shape[2] > 3:
                temp_image = temp_image[:, :, 0:3]
            skeletons_dict[file]['rgb_image'] = temp_image

        # 3. create a mask with the sensing areas for every image
        for key in skeletons_dict:
            rgb_temp = skeletons_dict[key]['rgb_image']

            # find grayscale and mask of the sensing areas
            temp_gray = ski.color.rgb2gray(rgb_temp)
            temp_mask_bool = np.invert(np.logical_not(temp_gray > 0.99))
            temp_mask_array = np.ones(temp_mask_bool.shape) * temp_mask_bool
            # save grayscale and mask
            skeletons_dict[key]['grayscale_image'] = temp_gray
            skeletons_dict[key]['sensing_area_mask'] = np.array(temp_mask_array)

        # 4. label sensing areas
        for key in skeletons_dict:
            mask_array_temp = skeletons_dict[key]['sensing_area_mask']
            labeled_array, df_circles = label_utils.get_area_labels(
                mask_input=mask_array_temp,
                n_sensing_areas=n_sensing_areas,
            )
            skeletons_dict[key]['labeled_array'] = labeled_array
            skeletons_dict[key]['df_circles'] = df_circles

            # save a mask per variable
            masks_per_variable_dict = {}
            for k_l, v_l in additional_specs['labels_to_sample_all'].items():
                temp_per_variable = np.zeros(labeled_array.shape)
                for i_l in v_l:
                    df_temp = df_circles[df_circles['sensing_area_name'] == i_l]
                    label_temp = int(df_temp['label'].to_numpy()[0])
                    temp_mask = np.invert(
                        np.logical_not(labeled_array == label_temp)
                    )
                    temp_per_variable = temp_per_variable + temp_mask
                masks_per_variable_dict[k_l] = temp_per_variable

            # create mask with all variables together
            n_vars = len(masks_per_variable_dict.keys())
            masks_per_variable = np.zeros((labeled_array.shape[0], labeled_array.shape[1], n_vars))
            for i_mv, e_mv in enumerate(masks_per_variable_dict.values()):
                masks_per_variable[:, :, i_mv] = e_mv

            masks_per_variable_dict['all_masks'] = masks_per_variable

            # return only the mask selected
            selected_var_mask = additional_specs['selected_mask']
            skeletons_dict[key]['masks_per_variable'] = masks_per_variable_dict[selected_var_mask]
        self.skeletons_dict = skeletons_dict

        # 4. load the label dictionary and calculate norm
        for idl in label_dictionary.keys():
            temp_array = label_dictionary[idl]['real']
            norm_temp_array = (temp_array - temp_array.min()) / (temp_array.max() - temp_array.min())
            label_dictionary[idl]['norm'] = (norm_temp_array + offset) / (1 + offset)
        self.label_dictionary = label_dictionary

        # 5. define total of images to generate
        self.n_synthetic_images = n_synthetic_images

        # 6. additional variables
        self.additional_specs = additional_specs
        self.target_image_shape = target_image_shape
        self.allow_incomplete = allow_incomplete
        self.incomplete_every_n = incomplete_every_n
        self.transform = transform
        self.illumination_noise = illumination_noise
        self.noise_level = noise_level
        self.masked_ground_truth = masked_ground_truth
        self.one_mask_input = one_mask_input

    def __len__(self):
        return self.n_synthetic_images

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # 1. choose random parameters
        #    td -> improve generalizability using just dictionaries
        #    random sample a parameter from each variable
        n_glucose_levels = len(self.label_dictionary['glucose']['real'])
        random_sample_glucose = np.random.randint(0, n_glucose_levels, size=1)

        n_ph_levels = len(self.label_dictionary['ph']['real'])
        random_sample_ph = np.random.randint(0, n_ph_levels, size=1)

        n_na_levels = len(self.label_dictionary['Na']['real'])
        random_sample_Na = np.random.randint(0, n_na_levels, size=1)

        n_temp_levels = len(self.label_dictionary['temperature']['real'])
        random_sample_temp = np.random.randint(0, n_temp_levels, size=1)

        rnd_parameter_sample = {
            'glucose': random_sample_glucose,
            'ph': random_sample_ph,
            'Na': random_sample_Na,
            'temperature': random_sample_temp,
        }

        # 2. sample pixels from the reference distributions
        empty_distributions_sampled = {}
        for ls_key, ls_item in self.additional_specs['labels_to_sample_all'].items():
            empty_distributions_sampled[ls_key] = {}
            list_of_repeated_areas = self.sensor_distributions[ls_key]['repeated_areas']
            if list_of_repeated_areas:
                empty_distributions_sampled[ls_key][ls_key] = {}
            else:
                for ls_subitem in ls_item:
                    empty_distributions_sampled[ls_key][ls_subitem] = {}

        #   sample pixels from the distribution to the mask
        for key in self.sensor_distributions:
            sample_param = rnd_parameter_sample[key][0]
            image_name_keys = list(self.sensor_distributions[key]['pixel_values'].keys())
            temp_arrays = self.sensor_distributions[key]['pixel_values'][image_name_keys[sample_param]]
            # if there are repeated areas concatenate distributions
            list_of_repeated_areas = self.sensor_distributions[key]['repeated_areas']
            if list_of_repeated_areas:
                sum_same_sensor = []
                for l_ra in list_of_repeated_areas:
                    temp_distr = temp_arrays[l_ra]
                    if len(sum_same_sensor) != 0:
                        sum_same_sensor = np.concatenate((sum_same_sensor, temp_distr), axis=1)
                    else:
                        sum_same_sensor = temp_distr
                empty_distributions_sampled[key][key] = sum_same_sensor
            # if not repeated then sample each sensing area separately
            else:
                for k_d in temp_arrays:
                    final_distr = temp_arrays[k_d]
                    empty_distributions_sampled[key][k_d] = final_distr

        # 3. allow for incomplete dataset
        #    a variable is removed with a chance of 1 every incomplete_every_n images
        random_sample_incomplete = np.random.randint(0, self.incomplete_every_n, size=1)
        labels_to_sample_all = self.additional_specs['labels_to_sample_all']
        complete_variable_list = list(labels_to_sample_all.keys())

        if self.allow_incomplete and random_sample_incomplete == 0:
            random_incomplete_variable = random.sample(complete_variable_list, 1)[0]
            new_variable_list = [x for x in complete_variable_list if x != random_incomplete_variable]
            labels_to_sample = {}
            for nv in new_variable_list:
                labels_to_sample[nv] = labels_to_sample_all[nv]
        else:
            labels_to_sample = labels_to_sample_all
            new_variable_list = list(labels_to_sample_all.keys())

        # 4. Sample pixels for each label

        #   sample pixels for each label
        sk_keys = list(self.skeletons_dict.keys())
        random_skeleton = np.random.randint(0, len(sk_keys), size=1)[0]
        sampled_skeleton = self.skeletons_dict[sk_keys[random_skeleton]]

        #   resize Images
        sampled_skeleton_rgb = ski.transform.resize(
            sampled_skeleton['rgb_image'],
            self.target_image_shape,
            anti_aliasing=True
        )
        sampled_labeled_array = ski.transform.resize(
            sampled_skeleton['labeled_array'],
            self.target_image_shape,
            order=0,
            anti_aliasing=False,
        )
        sampled_sensing_mask = ski.transform.resize(
            sampled_skeleton['masks_per_variable'],
            self.target_image_shape,
            order=0,
            anti_aliasing=False,
        )
        template_sensing_masks = np.zeros(
            sampled_skeleton_rgb.shape, dtype="uint8"
        )

        temp_df_circles = sampled_skeleton['df_circles']

        #   fill the sensing areas with sampled pixels
        for key, value in labels_to_sample.items():
            for lab_i in value:
                df_temp = temp_df_circles[temp_df_circles['sensing_area_name'] == lab_i]
                label_temp = int(df_temp['label'].to_numpy()[0])
                temp_mask = np.invert(
                    np.logical_not(sampled_labeled_array == label_temp)
                )

                sample_size = sum(temp_mask.flatten())
                if lab_i not in list(empty_distributions_sampled[key].keys()):
                    temp_variable_distribution = empty_distributions_sampled[key][key]
                else:
                    temp_variable_distribution = empty_distributions_sampled[key][lab_i]
                pixel_draw_pool = np.random.choice(temp_variable_distribution.shape[1], sample_size)

                # Sample every pixel from the distributions
                counter_pixels = 0
                for h_i in range(temp_mask.shape[0]):
                    for v_i in range(temp_mask.shape[1]):
                        if temp_mask[h_i, v_i]:
                            index_distribution = pixel_draw_pool[counter_pixels]
                            template_sensing_masks[h_i, v_i, :] = temp_variable_distribution[:, index_distribution]
                            counter_pixels += 1

        # 5. transform images to dtype
        image_template_sensing_masks = np.array(template_sensing_masks, dtype='uint8')

        single_mask = np.max(template_sensing_masks, axis=2)
        mask_areas_sampled = np.array(
            np.logical_not(single_mask == 0),
        )
        three_channel_mask = np.zeros(sampled_skeleton_rgb.shape, dtype='uint8')
        for i in range(3):
            three_channel_mask[:, :, i] = np.invert(mask_areas_sampled)
        sampled_skeleton_rgb_uint8 = np.array(sampled_skeleton_rgb * 255, dtype='uint8')
        skeleton_no_areas = np.array(sampled_skeleton_rgb_uint8 * three_channel_mask, dtype='uint8')
        skeleton_final = skeleton_no_areas + image_template_sensing_masks

        filtered_img = ski.filters.gaussian(skeleton_final, sigma=0.5, channel_axis=-1, mode='reflect')
        skeleton_final = np.array(filtered_img * 255, dtype='uint8')

        # fix mask shape
        if len(sampled_sensing_mask.shape) == 2:
            sampled_sensing_mask = np.expand_dims(sampled_sensing_mask, axis=2)

        # 6. Labels of sensing values
        image_label = {}
        for idr, vr in rnd_parameter_sample.items():
            sampled_idx = int(vr[0])
            if idr in new_variable_list:
                image_label[idr] = {
                    'real': self.label_dictionary[idr]['real'][sampled_idx],
                    'norm': self.label_dictionary[idr]['norm'][sampled_idx],
                    'complete': 1,
                }
            else:
                # if incomplete sample label is 0
                image_label[idr] = {
                    'real': 0,
                    'norm': 0,
                    'complete': 0,
                }

        # 7. Illumination Noise
        if self.illumination_noise:
            mask_sensor = skeleton_final[:, :, 0] == 0
            skeleton_noisy, sampled_kernel, noise_params = light_kernels.add_noise_to_image(
                image_in=skeleton_final,
                outer_mask_ref=mask_sensor,
                highest_noise_level=self.noise_level,
            )
        else:
            sampled_kernel = None
            noise_params = None

        # 8. Mask RGB original
        if self.masked_ground_truth:
            sampled_sensing_mask_all = np.sum(sampled_sensing_mask, axis=2)
            sampled_sensing_mask_all = np.expand_dims(sampled_sensing_mask_all, 2)
            triple_mask = np.concatenate(
                [sampled_sensing_mask_all, sampled_sensing_mask_all, sampled_sensing_mask_all],
                axis=2
            )
            skeleton_final = skeleton_final * triple_mask

        if self.one_mask_input:
            sampled_sensing_mask_all = np.sum(sampled_sensing_mask, axis=2)
            sampled_sensing_mask = np.expand_dims(sampled_sensing_mask_all, 2)

        # 9. Concatenate all
        selected_image_out = skeleton_final
        if self.illumination_noise:
            selected_image_out = skeleton_noisy

            # TODO idea to have kernel vals proportional to the noise
            # array_noise_vals = [
            #     noise_params['sampled_alpha'],
            #     noise_params['sampled_beta'],
            #     noise_params['sampled_gamma'],
            # ]
            # # idx_noise_max = np.argmax(np.abs(array_noise_vals))
            # # noise_max = array_noise_vals[idx_noise_max]
            # noise_max = np.max(np.abs(array_noise_vals))
            # sampled_kernel = sampled_kernel * (noise_max/255)
            # sampled_kernel = np.clip(sampled_kernel, 0, 1)

        sampled_kernel = np.expand_dims(sampled_kernel, axis=2)

        all_arrays_together = np.concatenate(
            [selected_image_out, sampled_sensing_mask, skeleton_final, sampled_kernel],
            axis=2
        )

        # reshape image channels first
        all_arrays_together = all_arrays_together.transpose((2, 0, 1))
        all_arrays_together = all_arrays_together.astype(np.float32)
        all_arrays_together = torch.as_tensor(all_arrays_together, dtype=torch.float32)

        # 10. Pytorch Transform Mask and Images
        if self.transform:
            all_arrays_together = self.transform(all_arrays_together)

        n_masks = sampled_sensing_mask.shape[2] + 3
        image_masks = all_arrays_together[0:n_masks, :, :]
        original_image = all_arrays_together[n_masks:n_masks+3, :, :]
        sampled_kernel = all_arrays_together[n_masks+3:n_masks+4, :, :]

        return {
            "image_and_masks": image_masks,
            "original_image": original_image,
            "light_kernel": sampled_kernel,
            "label_sensor": image_label,
            "label_noise": noise_params,
        }


if __name__ == '__main__':
    TEST_DATA_LOADER = True

    if TEST_DATA_LOADER:
        # Config
        ROOT_PATH = '/Users/castelblanco/Documents/PhD-Repos/neonatal_sensors/multiarray_color_sensors'
        PATH_IMAGE_SKELETONS = 'data/processed/2024_sensor_skeletons/'
        PATHS_LIST_REF_COLOR_DICT = {
            'glucose': 'data/processed/2024_glucose_ref/glucose_ref_pixels_2025_01_29.pkl',
            'ph': 'data/processed/2024_ph_ref/ph_ref_pixels_2024_07_30.pkl',
            'Na': 'data/processed/2024_na_ref/sodium_ref_pixels_2025_01_29.pkl',
            'temperature': 'data/processed/2024_temp_ref/temp_ref_pixels_2024_08_20.pkl'
        }
        LABEL_DICTIONARY = {
            'glucose': {
                'real': np.array([0, 0.04, 0.08, 0.16, 0.3125, 0.625]),
            },
            'ph': {
                'real': np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            },
            'Na': {
                'real': np.array([0, 2.0, 4.5, 8.0, 20]),
            },
            'temperature': {
                'real': np.arange(32, 41, 0.5),
            },
        }

        N_SYNTHETIC_IMAGES = 10
        TARGET_IMAGE_SHAPE = (256, 256)

        # additional specs
        LABELS_TO_SAMPLE_ALL = {
            'glucose': ['g_ul', 'g_ur', 'g_d'],
            'ph': ['ph_ul', 'ph_ur', 'ph_d'],
            'Na': ['Na'],
            'temperature': ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
        }

        SELECTED_MASK = 'all_masks'  # 'all' , 'glucose', 'ph'

        LIST_REPEATED_AREAS = {
            'glucose': ['g_ul', 'g_ur', 'g_d'],
            'ph': None,
            'Na': None,
            'temperature': None,
        }

        ADDITIONAL_SPECS = {
            'labels_to_sample_all': LABELS_TO_SAMPLE_ALL,
            'list_repeated_areas': LIST_REPEATED_AREAS,
            'selected_mask': SELECTED_MASK,
        }

        OFFSET = 0.1

        MASKED_OUTPUT = False  # True
        ONE_MASK_INPUT = True

        # Create Data Transforms
        composed_transforms = transforms.Compose([
            # rotation
            transforms.RandomRotation(
                degrees=(-20, 20)
            ),
            # perspective transforms
            transforms.RandomPerspective(
                distortion_scale=0.6, # 0.2
                p=0.5,
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            # resized crop
            transforms.RandomResizedCrop(
                size=TARGET_IMAGE_SHAPE[0],
                scale=(0.8, 1.3),
                ratio=(0.8, 1.3),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
        ])

        # Dataset Class Object
        train_set = DatasetSyntheticSensors(
            root_path=ROOT_PATH,
            paths_list_ref_color_dict=PATHS_LIST_REF_COLOR_DICT,
            path_image_skeletons=PATH_IMAGE_SKELETONS,
            label_dictionary=LABEL_DICTIONARY,
            additional_specs=ADDITIONAL_SPECS,
            n_synthetic_images=N_SYNTHETIC_IMAGES,
            target_image_shape=TARGET_IMAGE_SHAPE,
            n_sensing_areas=14,
            image_skeletons_format='.png',
            allow_incomplete=False,
            incomplete_every_n=10,
            transform=composed_transforms,
            illumination_noise=True,
            noise_level=0.5,
            offset=OFFSET,
            masked_ground_truth=MASKED_OUTPUT,
            one_mask_input=ONE_MASK_INPUT,
        )

        # Dataloader
        train_dataloader = DataLoader(
            train_set,
            batch_size=4,
            shuffle=True
        )

        # Display image and label
        sample_batch = next(iter(train_dataloader))
        noisy_image_plus_masks_batch = sample_batch['image_and_masks']
        labels_batch = sample_batch['label_sensor']
        label_noise_batch = sample_batch['label_noise']
        original_image_batch = sample_batch['original_image']
        light_kernel_batch = sample_batch['light_kernel']

        for i in range(4):
            temp_image_r = noisy_image_plus_masks_batch[i, 0, :, :].numpy().astype('uint8')
            temp_image_g = noisy_image_plus_masks_batch[i, 1, :, :].numpy().astype('uint8')
            temp_image_b = noisy_image_plus_masks_batch[i, 2, :, :].numpy().astype('uint8')

            img_temp = np.zeros((TARGET_IMAGE_SHAPE[0], TARGET_IMAGE_SHAPE[1], 3), dtype='uint8')
            img_temp[:, :, 0] = temp_image_r
            img_temp[:, :, 1] = temp_image_g
            img_temp[:, :, 2] = temp_image_b

            plt.imshow(img_temp)
            plt.show()

        for i in range(4):
            temp_image_r = original_image_batch[i, 0, :, :].numpy().astype('uint8')
            temp_image_g = original_image_batch[i, 1, :, :].numpy().astype('uint8')
            temp_image_b = original_image_batch[i, 2, :, :].numpy().astype('uint8')

            img_temp = np.zeros((TARGET_IMAGE_SHAPE[0], TARGET_IMAGE_SHAPE[1], 3), dtype='uint8')
            img_temp[:, :, 0] = temp_image_r
            img_temp[:, :, 1] = temp_image_g
            img_temp[:, :, 2] = temp_image_b

            plt.imshow(img_temp)
            plt.show()

        for i in range(4):
            temp_image = light_kernel_batch[i, 0, :, :].numpy()
            plt.imshow(temp_image)
            plt.show()

        if ONE_MASK_INPUT:
            for i in range(4):
                plt.imshow(noisy_image_plus_masks_batch[i, 3, :, :])
                plt.show()
        else:
            for i in range(4):
                fig, axs = plt.subplots(2, 2)
                axs[0, 0].imshow(noisy_image_plus_masks_batch[i, 3, :, :])
                axs[0, 1].imshow(noisy_image_plus_masks_batch[i, 4, :, :])
                axs[1, 0].imshow(noisy_image_plus_masks_batch[i, 5, :, :])
                axs[1, 1].imshow(noisy_image_plus_masks_batch[i, 6, :, :])
                plt.show()
