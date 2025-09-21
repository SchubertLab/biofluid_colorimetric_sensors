import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from scipy.optimize import curve_fit


# We fit a curve of the form
# y = A + B log x
def logarithmic_model(x, a, b):
    return a * np.log(x) + b


def exponential_model(y, a, b):
    return np.exp(np.divide((y - b), a))


# Load the pixel values per sensor per level
def load_pixel_value_list(dictionary_in, sample_channel=1):
    data_pixels = []
    for image_temp in dictionary_in.keys():
        temp_plot_sample = dictionary_in[image_temp]
        array_same_sensor = np.array([])
        for k_temp in temp_plot_sample.keys():
            temp_array = temp_plot_sample[k_temp][sample_channel]
            array_same_sensor = np.concatenate(
                (array_same_sensor, temp_array),
                axis=None
            )
        data_pixels.append(array_same_sensor)
    return data_pixels


def get_percentiles_of_lists(list_array, percentiles_list_array):
    ref_pixels_within_25_75_perc = []
    for idx, pix_per_image in enumerate(list_array):
        temp_list = list(
            filter(lambda x: percentiles_list_array[idx][1] > x > percentiles_list_array[idx][0], pix_per_image))
        ref_pixels_within_25_75_perc.append(temp_list)
    return ref_pixels_within_25_75_perc


# ------------------------------------------------------------------------------
# Load Data
ROOT_DIR = '/Users/castelblanco/Documents/PhD-Repos/neonatal_sensors/colorimetric_sensors/Colorimetric_Sensors/data/processed/'
REF_DIR = '2024_references/'
REF_FILENAME = 'glucose_ref_pixels_2024_07_02.pkl'
SALIVA_DIR = '2024_saliva_exp/saliva_experiment/AC/'
SALIVA_FILENAME = 'glucose_saliva_pixels_2024_07_02.pkl'
PATH_ELISA = ROOT_DIR + SALIVA_DIR + 'elisa_sensor_results_no_outliers.csv'  # elisa_sensor_results

with open(ROOT_DIR + REF_DIR + REF_FILENAME, 'rb') as file:
    dict_reference_val_per_image = pickle.load(file)

with open(ROOT_DIR + SALIVA_DIR + SALIVA_FILENAME, 'rb') as file:
    dict_saliva_val_per_image = pickle.load(file)

# Load the pixel values per sensor per level
ref_pixels = load_pixel_value_list(
    dictionary_in=dict_reference_val_per_image,
    sample_channel=1,  # Green
)

# filter data within the quantile [0.25-0.75]
ref_percentiles_values = [np.percentile(x, [25, 75], axis=0) for x in ref_pixels]
ref_pixels_within_25_75_perc = get_percentiles_of_lists(
    list_array=ref_pixels,
    percentiles_list_array=ref_percentiles_values,
)

# Assign dictionary per label
data_for_regression = ref_pixels_within_25_75_perc
levels_glucose = [1e-3, 0.0391, 0.0781, 0.15625, 0.25, 0.45, 0.98, 2.0]
data_x = []
data_y = []
for idx, pix_per_image in enumerate(data_for_regression):
    data_y = np.concatenate((data_y, pix_per_image), axis=None)
    temp_level_array = levels_glucose[idx] * np.ones(len(pix_per_image))
    data_x = np.concatenate((data_x, temp_level_array), axis=None)

# Use curve_fit to fit the logarithmic model to the data
popt, pcov = curve_fit(logarithmic_model, data_x, data_y)

# Extract the optimal parameters
a_opt, b_opt = popt
print(f"Optimal parameters: a = {a_opt}, b = {b_opt}")

# Model Prediction

# load experimental data
experiment_pixels = load_pixel_value_list(
    dictionary_in=dict_saliva_val_per_image,
    sample_channel=1,  # Green
)
# filter data within the quantile [0.25-0.75]
experiment_percentiles_values = [np.percentile(x, [25, 75], axis=0) for x in experiment_pixels]
experiment_pixels_within_25_75_perc = get_percentiles_of_lists(
    list_array=experiment_pixels,
    percentiles_list_array=experiment_percentiles_values,
)
# predict for every list
x_experiment_predictions_per_image = []
for temp_exp in experiment_pixels_within_25_75_perc:
    temp_pred = [exponential_model(p, a_opt, b_opt) for p in temp_exp]
    x_experiment_predictions_per_image.append(temp_pred)

mean_preds_per_image = [np.mean(x) for x in x_experiment_predictions_per_image]
std_preds_per_image = [np.std(x) for x in x_experiment_predictions_per_image]

# ------------------------------------------------------------------------------
# PLOTS REFERENCES
fig, ax1 = plt.subplots(1, 1, figsize=(12 * (0.6), 6 * (0.6)))
ax1.set_title('Glucose Levels Colorimetric Response \n (x3 sensors)')
ax1.set_ylabel('Green Channel')
ax1.set_xlabel('Glucose Level [mg/mL]')

labels_temp = ['G=0', 'G=0.04', 'G=0.08', 'G=0.16', 'G=0.25', 'G=0.45', 'G=1.0', 'G=2.0']
temp_vplot = ax1.violinplot(
    ref_pixels,
    levels_glucose,
    showmedians=True,
    showextrema=False,
    widths=0.06,
)
ax1.scatter(data_x, data_y, label='Data [0.25-0.75]', marker='.', alpha=0.1, color='cornflowerblue')
x_log_fx = np.linspace(1e-3, levels_glucose[-1], 100)
ax1.plot(
    x_log_fx, logarithmic_model(x_log_fx, *popt),
    linewidth=2,
    color='cornflowerblue',
    label='Fitted Log. Curve',
)
pdf_filename = 'Glucose_Levels_Distribution_Fitted.pdf'
plt.tight_layout()
#plt.savefig(ROOT_DIR + pdf_filename, format='pdf')
plt.show()

# ------------------------------------------------------------------------------
# PLOTS EXPERIMENT
scale_factor = 0.8
fig2, ax2 = plt.subplots(1, 1, figsize=(12 * scale_factor, 6 * scale_factor))

ax2.violinplot(
    experiment_pixels,
    mean_preds_per_image,
    showmedians=True,
    showextrema=False,
    widths=0.01,
)
x_log_fx2 = np.linspace(1e-3, 0.2, 50)
ax2.plot(
    x_log_fx2, logarithmic_model(x_log_fx2, *popt),
    linewidth=2,
    color='cornflowerblue',
    label='Fitted Log. Curve',
)
ax2.legend()
plt.show()



