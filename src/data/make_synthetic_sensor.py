import os
import random
import pickle
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import src.utils.label_areas_utils as label_utils

from skimage.transform import resize

# Constants
N_SENSING_AREAS = 14
IMAGE_SKELETONS_PATH = '../../data/processed/2024_sensor_skeletons/'
IMAGES_FORMAT = '.png'

LABEL_DICTIONARY = {
    'glucose': {
        'real': np.array([0, 0.04, 0.08, 0.16, 0.25, 0.45, 1]),
    },
    'ph': {
        'real': np.array([3, 4, 5, 6, 7, 8, 9]),
    },
    'Na': {
        'real': np.array([0, 4.5, 6, 8, 11.5, 20]),
    },
    'temperature': {
        'real': np.arange(29, 42, 0.5),
    },
}

for idl in LABEL_DICTIONARY.keys():
    temp_array = LABEL_DICTIONARY[idl]['real']
    norm_temp_array = (temp_array - temp_array.min()) / (temp_array.max() - temp_array.min())
    LABEL_DICTIONARY[idl]['norm'] = norm_temp_array


# Load distributions for Pixels
sensor_distributions = {
    'glucose': {},
    'ph': {},
    'Na': {},
    'temperature': {},
}

REF_PIXEL_DISTRIBUTIONS = [
    '../../data/processed/2024_glucose_ref/glucose_ref_pixels_2024_07_02.pkl',
    '../../data/processed/2024_ph_ref/ph_ref_pixels_2024_07_30.pkl',
    '../../data/processed/2024_na_ref/sodium_ref_pixels_2024_08_12.pkl',
    '../../data/processed/2024_temp_ref/temp_ref_pixels_2024_08_20.pkl'
]

LABELS_TO_SAMPLE_ALL = {
    'glucose': ['g_ul', 'g_ur', 'g_d'],
    'ph': ['ph_ul', 'ph_ur', 'ph_d'],
    'Na': ['Na'],
    'temperature': ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
}

LIST_REPEATED_AREAS = [['g_ul', 'g_ur', 'g_d'], None, None, None]

empty_distributions_sampled = {
    'glucose': {
    },
    'ph': {
        'ph_ul': {},
        'ph_ur': {},
        'ph_d': {},
    },
    'Na': {
    },
    'temperature': {
        'T0': {},
        'T1': {},
        'T2': {},
        'T3': {},
        'T4': {},
        'T5': {},
        'T6': {},
    }
}

additional_specs = {
    'sensor_distributions': sensor_distributions,
    'labels_to_sample_all': LABELS_TO_SAMPLE_ALL,
    'list_repeated_areas': LIST_REPEATED_AREAS,
    'empty_distributions_sampled': empty_distributions_sampled,
}

IMAGE_GENERATION_SHAPE = (250, 250)

# Allow generation of sensors where the variables are not complete (every 1/N chance of incomplete)
GENERATE_INCOMPLETE = True
N_RATIO_INCOMPLETE = 10

# -------------------------------------------------------------------------
# Load the pixel distributions per variable
d_count = 0
for key in sensor_distributions:
    pixel_path = REF_PIXEL_DISTRIBUTIONS[d_count]
    with open(pixel_path, 'rb') as file:
        dictionary_values_per_image = pickle.load(file)
    sensor_distributions[key]['pixel_values'] = dictionary_values_per_image
    sensor_distributions[key]['repeated_areas'] = LIST_REPEATED_AREAS[d_count]
    d_count += 1

# -------------------------------------------------------------------------
# Load skeletons in the folder
file_names = sorted(
    [fname for fname in os.listdir(IMAGE_SKELETONS_PATH) if fname.endswith(IMAGES_FORMAT)]
)

skeletons_dict = {}
for file in file_names:
    skeletons_dict[file] = {
        'rgb_image': [],
        'grayscale_image': [],
        'sensing_area_mask': [],
        'df_circles': [],
        'labeled_array': [],
    }
    temp_file_path = os.path.join(IMAGE_SKELETONS_PATH, file)
    image_name_temp = file.split('/')[-1]
    temp_image = ski.io.imread(temp_file_path)
    if temp_image.shape[2] > 3:
        temp_image = temp_image[:, :, 0:3]
    skeletons_dict[file]['rgb_image'] = temp_image

# Create a mask with the sensing areas for every image
for key in skeletons_dict:
    rgb_temp = skeletons_dict[key]['rgb_image']

    # find grayscale and mask of the sensing areas
    temp_gray = ski.color.rgb2gray(rgb_temp)
    temp_mask_bool = np.invert(np.logical_not(temp_gray > 0.99))
    temp_mask_array = np.ones(temp_mask_bool.shape) * temp_mask_bool
    # save grayscale and mask
    skeletons_dict[key]['grayscale_image'] = temp_gray
    skeletons_dict[key]['sensing_area_mask'] = np.array(temp_mask_array)


# -------------------------------------------------------------------------
# Label Sensing Areas
for key in skeletons_dict:
    mask_array_temp = skeletons_dict[key]['sensing_area_mask']
    labeled_array, df_circles = label_utils.get_area_labels(
        mask_input=mask_array_temp,
        n_sensing_areas=N_SENSING_AREAS,
    )
    skeletons_dict[key]['labeled_array'] = labeled_array
    skeletons_dict[key]['df_circles'] = df_circles

key_temp = list(skeletons_dict.keys())[0]
temp_img = skeletons_dict[key_temp]['labeled_array']

# -------------------------------------------------------------------------
# Random sample a parameter from each variable
n_glucose_levels = len(LABEL_DICTIONARY['glucose']['real'])
random_sample_glucose = np.random.randint(0, n_glucose_levels, size=1)

n_ph_levels = len(LABEL_DICTIONARY['ph']['real'])
random_sample_ph = np.random.randint(0, n_ph_levels, size=1)

n_na_levels = len(LABEL_DICTIONARY['Na']['real'])
random_sample_Na = np.random.randint(0, n_na_levels, size=1)

n_temp_levels = len(LABEL_DICTIONARY['temperature']['real'])
random_sample_temp = np.random.randint(0, n_temp_levels, size=1)

rnd_parameter_sample = {
    'glucose': random_sample_glucose,
    'ph': random_sample_ph,
    'Na': random_sample_Na,
    'temperature': random_sample_temp,
}

# Sample pixels from the distribution to the mask
for key in sensor_distributions:
    sample_param = rnd_parameter_sample[key][0]
    image_name_keys = list(sensor_distributions[key]['pixel_values'].keys())
    temp_arrays = sensor_distributions[key]['pixel_values'][image_name_keys[sample_param]]
    # if repeated areas concatenate distributions
    list_of_repeated_areas = sensor_distributions[key]['repeated_areas']
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
            empty_distributions_sampled[key][k_d] = temp_arrays[k_d]


# --------------------------------------------------------------------------------------------
# Remove variables to allow for sensors with not all variables printed

# A variable is removed with a chance of 1 every N_RATIO_INCOMPLETE images
random_sample_incomplete = np.random.randint(0, N_RATIO_INCOMPLETE, size=1)
complete_variable_list = list(LABELS_TO_SAMPLE_ALL.keys())
if GENERATE_INCOMPLETE and random_sample_incomplete == 0:
    random_incomplete_variable = random.sample(complete_variable_list, 1)[0]
    new_variable_list = [x for x in complete_variable_list if x != random_incomplete_variable]
    labels_to_sample = {}
    for nv in new_variable_list:
        labels_to_sample[nv] = LABELS_TO_SAMPLE_ALL[nv]
else:
    labels_to_sample = LABELS_TO_SAMPLE_ALL
    new_variable_list = list(LABELS_TO_SAMPLE_ALL.keys())

# --------------------------------------------------------------------------------------------
# Sample pixels for each label
random_skeleton = np.random.randint(0, 12, size=1)[0]
sk_keys = list(skeletons_dict.keys())
sampled_skeleton = skeletons_dict[sk_keys[random_skeleton]]

# Resize Images
sampled_skeleton_rgb = resize(
    sampled_skeleton['rgb_image'],
    IMAGE_GENERATION_SHAPE,
    anti_aliasing=True
)
sampled_labeled_array = resize(
    sampled_skeleton['labeled_array'],
    IMAGE_GENERATION_SHAPE,
    order=0,
    anti_aliasing=False,
)
sampled_sensing_mask = resize(
    sampled_skeleton['sensing_area_mask'],
    IMAGE_GENERATION_SHAPE,
    order=0,
    anti_aliasing=False,
)

template_sensing_masks = np.zeros(
    sampled_skeleton_rgb.shape, dtype="uint8"
)

temp_df_circles = sampled_skeleton['df_circles']

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


image_template_sensing_masks = np.array(template_sensing_masks, dtype="uint8")

single_mask = np.max(template_sensing_masks, axis=2)
mask_areas_sampled = np.array(
    np.logical_not(single_mask == 0),
)
three_channel_mask = np.zeros(sampled_skeleton_rgb.shape, dtype="uint8")
for i in range(3):
    three_channel_mask[:, :, i] = np.invert(mask_areas_sampled)

sampled_skeleton_rgb_uint8 = np.array(sampled_skeleton_rgb*255, dtype="uint8")

skeleton_no_areas = np.array(sampled_skeleton_rgb_uint8*three_channel_mask, dtype="uint8")

skeleton_final = skeleton_no_areas + image_template_sensing_masks

filtered_img = ski.filters.gaussian(skeleton_final, sigma=0.5, channel_axis=-1, mode='reflect')
skeleton_final_uint8 = np.array(filtered_img*255, dtype="uint8")

# --------------------------------------------------------------------------------------------
# Plots

plt.imshow(skeleton_final_uint8)
plt.show()

plt.imshow(skeleton_no_areas)
plt.show()

plt.imshow(image_template_sensing_masks)
plt.show()

# Labeled Array Plot
plt.imshow(sampled_skeleton['labeled_array'])
for index, row in temp_df_circles.iterrows():
    y = row['centroid-0'] - (sampled_skeleton['labeled_array'].shape[0]*0.03)
    x = row['centroid-1'] - (sampled_skeleton['labeled_array'].shape[0]*0.06)
    s = row['sensing_area_name']
    plt.text(x, y, s, fontsize=8, color='white')
plt.show()

# print the label of the output:
image_label = {}

for idr, vr in rnd_parameter_sample.items():
    sampled_idx = int(vr[0])
    if idr in new_variable_list:
        image_label[idr] = {
            'real': LABEL_DICTIONARY[idr]['real'][sampled_idx],
            'norm': LABEL_DICTIONARY[idr]['norm'][sampled_idx],
            'complete': True,
        }
    else:
        # if incomplete sample label is none
        image_label[idr] = {
            'real': np.nan,
            'norm': np.nan,
            'complete': False,
        }
print(image_label)

# TODO: make a package of images to export (RGB and the masks per variable and total mask), the label is there.
