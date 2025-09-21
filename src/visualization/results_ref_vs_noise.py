import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT_PATH = './multiarray_color_sensors/'

model_eval_results_path = ROOT_PATH + 'results/single_var_models/ph/predictions_1.csv'
df_results = pd.read_csv(model_eval_results_path)
target_array = df_results['target_0_1'].to_numpy()
pred_array = df_results['pred_0_1'].to_numpy()
pred_array_real = df_results['pred_array_real'].to_numpy()
target_array_real = df_results['target_array_real'].to_numpy()

noise_eval_results_path = ROOT_PATH + 'results/from_unet_mlp_results/ph/predictions_noise_1.csv'
df_noise = pd.read_csv(noise_eval_results_path)
noise_target_array = df_noise['target_0_1'].to_numpy()
noise_pred_array = df_noise['pred_0_1'].to_numpy()
noise_pred_array_real = df_noise['pred_array_real'].to_numpy()
noise_target_array_real = df_noise['target_array_real'].to_numpy()


OFFSET = False
if OFFSET:
    offset = 0
    scale = 1
    pred_array_offset = (pred_array_real * scale) + offset
    df_results['pred_array_offset'] = pred_array_offset

df_all = df_noise
df_all['white_pred_real_0_1'] = df_results['pred_0_1'][0:120]
df_all['white_target_real_0_1'] = df_results['target_0_1'][0:120]

df_all['pred_0_1'] = df_results['pred_0_1']*1.01

fig = plt.figure(figsize=(5, 3))
sns.boxplot(
    data=df_all,
    x="white_target_real_0_1",
    y="white_pred_real_0_1",
    # width=1.1
    color='lightskyblue',
)

sns.swarmplot(
    data=df_all,
    x="white_target_real_0_1",
    y="white_pred_real_0_1",
    # width=1.1
    color='gray',
)

fig.tight_layout()
plt.xlabel('pH')
plt.ylabel('pH Model Prediction')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], [0, 3,4,5,6,7,8,9])
plt.show()
