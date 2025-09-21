import torch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

import src.test.denoise_unet_mlp_25.config_eval_unet_mlp_25 as config_args

import src.models.model_denoising_unet_mlp_25 as model_denoising_mlp
import src.train.denoise_multiple_var_prediction.load_noise_dataset as sensor_noise_dataset


def evaluate(config, denoising_mlp_model, device, test_loader,
             denoising_loss_dict, mlp_loss, epoch=1, weights=None):
    if weights is None:
        weights = [0.07, 0.07, 0.07, 0.07, 2.6]

    # Unet model prediction
    denoising_mlp_model.eval()
    # Arrays of individual predictions
    pred_array = []
    target_array = []

    with torch.no_grad():
        # Train per batch
        for batch_idx, sample_batch in enumerate(test_loader):

            # Sampled images
            noisy_image_plus_masks_batch = sample_batch['image_and_masks']
            original_image_batch = sample_batch['original_image']
            light_kernel_batch = sample_batch['light_kernel']

            original_imag_plus_kernel = torch.cat(
                (original_image_batch, light_kernel_batch),
                dim=1
            )

            # Sampled labels
            labels_batch = sample_batch['label_sensor']
            label_noise_batch = sample_batch['label_noise']

            # Data to device
            input_data = noisy_image_plus_masks_batch.to(device)
            original_imag_plus_kernel = original_imag_plus_kernel.to(device)

            # Prediction denoising + mlp model (features and output)
            # start_time_1 = time.time()
            output_prediction = denoising_mlp_model(input_data)
            # end_time_1 = time.time()
            # print(f"Section 1 execution time: {end_time_1 - start_time_1:.6f} seconds")

            output_latent_features = output_prediction[0]  # Latent Features
            output_img_kernel = output_prediction[1]  # RGB image plus kernel
            output_variable = output_prediction[2]  # Label

            # Prediction and Loss for variables
            mae_fx = torch.nn.L1Loss(reduction='none')

            # HERE
            # Denoising Losses
            original_image_batch_norm = torch.div(original_image_batch, 255)
            output_img_kernel_rgb = output_img_kernel[:, 0:3, :, :]
            denoising_loss_red = denoising_loss_dict['rgb'](
                output_img_kernel_rgb[:, 0, :, :].to(device),
                original_image_batch_norm[:, 0, :, :].to(device)
            ).to(device)
            denoising_loss_green = denoising_loss_dict['rgb'](
                output_img_kernel_rgb[:, 1, :, :].to(device),
                original_image_batch_norm[:, 1, :, :].to(device),
            ).to(device)
            denoising_loss_blue = denoising_loss_dict['rgb'](
                output_img_kernel_rgb[:, 2, :, :].to(device),
                original_image_batch_norm[:, 2, :, :].to(device),
            ).to(device)

            output_img_kernel_1 = torch.unsqueeze(
                output_img_kernel[:, 3, :, :], dim=1
            ).to(device)

            denoising_loss_kernel = denoising_loss_dict['kernel'](
                output_img_kernel_1,
                light_kernel_batch.to(device)
            ).to(device)

            # MLP Loss
            name_variable = config['variable_pred_mlp_only']
            label_variable = labels_batch[name_variable]['norm'].to(device)
            label_variable = (torch.unsqueeze(label_variable, dim=0))
            label_variable = label_variable.permute(1, 0).to(device)
            loss_variable_mlp = mlp_loss(
                output_variable, label_variable
            ).to(device)

            denoising_loss_rgbk = denoising_loss_red + denoising_loss_green + denoising_loss_blue + denoising_loss_kernel

            denoising_loss_test = torch.stack([
                denoising_loss_red * weights[0],
                denoising_loss_green * weights[1],
                denoising_loss_blue * weights[2],
                denoising_loss_kernel * weights[3],
                loss_variable_mlp * weights[4],
            ], dim=0)

            total_val_loss = denoising_loss_test.sum()

            print('epoch', epoch)

            mae_variable = mae_fx(
                output_variable, label_variable
            ).to(device)

            # save outputs to plot
            pred_array.append(output_variable.cpu().detach().numpy())
            target_array.append(label_variable.cpu().detach().numpy())

            # Add MAE for all samples in the batch
            if batch_idx == 0:
                array_mae_variable = torch.flatten(mae_variable)
            else:
                array_mae_variable = torch.cat((array_mae_variable, torch.flatten(mae_variable)), dim=0)

            print('epoch', epoch)

            if batch_idx % config['log_interval'] == 0:
                print('Val. Epoch: {} [{}/{} ({:.0f}%)]\t'.format(
                    epoch, batch_idx * noisy_image_plus_masks_batch.shape[0], len(test_loader.dataset),
                           100. * batch_idx / len(test_loader)))

                print('Variable: {}, Val. Total Loss:{:.6f}'.format(name_variable, total_val_loss.item()))
                print('Variable: {}, Val. MLP Loss:{:.6f}'.format(name_variable, loss_variable_mlp.item()))

        mae_results_per_epoch = torch.div(array_mae_variable.sum(), len(array_mae_variable - 1))
    return mae_results_per_epoch, pred_array, target_array


def lin_reg_norm_to_real(unique_norm, unique_real):
    slope = (np.max(unique_real) - np.min(unique_real)) / (np.max(unique_norm) - np.min(unique_norm))
    intercept = np.max(unique_real) - (1 * slope)
    return slope, intercept


def norm_to_real_units(input_array, slope, intercept):
    output_array = np.zeros(len(input_array))
    for i, x in enumerate(input_array):
        if x != 0:
            output_array[i] = slope * x + intercept
    output_array = np.round(output_array, 3)
    return output_array


# --------------------------------------------------------------------------------------------------------------------
# Load Previous Results
LOAD_PREV_RESULTS = True

# Load Configuration
config = config_args.main()

# Values of the real scale
TARGET_VAR = config['variable_pred_mlp_only']
real_unique_array = np.array(config['label_dictionary'][TARGET_VAR]['real'])

if not LOAD_PREV_RESULTS:
    SEED = config['seed']
    torch.manual_seed(SEED)

    if config['use_cuda'] and torch.cuda.is_available():
        print('cuda available:', torch.cuda.is_available())
        device = torch.device("cuda")
        torch.backends.cudnn.enabled = False  # (needed for Denbi?)
        cuda_kwargs = {
            'batch_size': config['batch_size'],
            'num_workers': 1,
            # 'pin_memory': False,
            # 'shuffle': True
        }

    else:
        device = torch.device("cpu")

        cuda_kwargs = {
            'batch_size': config['batch_size']
        }
    print('device:', device)

    # Load Dataset
    _, test_dataloader = sensor_noise_dataset.load_noise_dataset(config)

    # Load Model
    # Load UNET Model
    dae_model_config = config['dae_model_config']

    dae_mlp_model = model_denoising_mlp.DenoisingUnetMLP(
        n_in_channels=dae_model_config['n_in_channels'],
        n_out_channels=dae_model_config['n_out_channels'],
        init_features=dae_model_config['init_features'],
        kernel=2,
        stride=2,
    )

    dae_mlp_model = dae_mlp_model.to(device)
    print(dae_mlp_model)

    # Load Unet Pretrained Weights
    dae_mlp_model_saved_path = config['dae_model_path']
    checkpoint = torch.load(
        dae_mlp_model_saved_path,
        map_location=torch.device('cpu'),
        weights_only=True,
    )
    dae_mlp_model.load_state_dict(checkpoint)

    # Loss function
    # Denoising Autoencoder Loss
    dae_loss_dict = {
        'kernel': None,
        'rgb': None,
    }
    if config['dae_loss_type'] == 'CE':
        dae_loss_dict['rgb'] = torch.nn.CrossEntropyLoss(reduction='mean')
        dae_loss_dict['kernel'] = torch.nn.CrossEntropyLoss(reduction='mean')
    elif config['dae_loss_type'] == 'BCE':
        dae_loss_dict['rgb'] = torch.nn.BCEWithLogitsLoss(reduction='mean')
        dae_loss_dict['kernel'] = torch.nn.BCEWithLogitsLoss(reduction='mean')
    elif config['dae_loss_type'] == 'MSE':
        dae_loss_dict['rgb'] = torch.nn.MSELoss(reduction='mean')
        dae_loss_dict['kernel'] = torch.nn.MSELoss(reduction='mean')
    elif config['dae_loss_type'] == 'BCE/MSE':
        dae_loss_dict['rgb'] = torch.nn.BCEWithLogitsLoss(reduction='mean')
        dae_loss_dict['kernel'] = torch.nn.MSELoss(reduction='mean')

    # MLP Loss function
    if config['mlp_loss_type'] == 'L1':
        label_loss = torch.nn.L1Loss(reduction='mean')
    elif config['mlp_loss_type'] == 'MSE':
        label_loss = torch.nn.MSELoss(reduction='mean')
    else:
        label_loss = None

    mae_results_per_epoch, pred_array, target_array = evaluate(
        config=config,
        denoising_mlp_model=dae_mlp_model,
        device=device,
        test_loader=test_dataloader,
        denoising_loss_dict=dae_loss_dict,
        mlp_loss=label_loss,
        epoch=1
    )

    pred_array = np.array(pred_array).flatten()
    target_array = np.array(target_array).flatten()

    # Real units for predictions ------------------------------------------------------------------------------------
    # Slope and intersect to translate units

    unique_norm = np.unique(target_array)

    var_slope, var_intercept = lin_reg_norm_to_real(
        unique_norm=unique_norm,
        unique_real=real_unique_array,
    )
    # Target array to real units
    target_array_real = norm_to_real_units(
        input_array=target_array,
        slope=var_slope,
        intercept=var_intercept,
    )

    # Pred array to real units
    pred_array_real = norm_to_real_units(
        input_array=pred_array,
        slope=var_slope,
        intercept=var_intercept,
    )

    # Dataframe for results
    dict_results = {
        'target_0_1': target_array,
        'pred_0_1': pred_array,
        'target_array_real': target_array_real,
        'pred_array_real': pred_array_real,
    }

    df_results = pd.DataFrame.from_dict(dict_results)

    # Save df with results
    if config['save_eval_results']:
        df_results.to_csv(
            config['model_eval_results_path'],
            index=False
        )

else:
    df_results = pd.read_csv(config['model_eval_results_path'])
    target_array = df_results['target_0_1'].to_numpy()
    pred_array = df_results['pred_0_1'].to_numpy()

# Metrics ------------------------------------------------------------------------------------------------------------
# Metrics with 0-1 scale

MAE_ANALYSIS = True

if MAE_ANALYSIS:

    # Find MAE and MSE
    MSE_final = np.divide(np.sum((pred_array - target_array) ** 2), len(target_array))
    MAE_final = np.divide(np.sum(np.abs(pred_array - target_array)), len(target_array))
    print('MAE_norm:', np.round(MAE_final, 5))
    print('MSE_norm:', np.round(MSE_final, 5))

    # Fit Linear model with zero
    pred_sck = pred_array.reshape(-1, 1)
    target_sck = target_array.reshape(-1, 1)
    lin_model = LinearRegression()
    lin_model.fit(pred_sck, target_sck)
    print('R_norm: ')
    print(lin_model.score(target_sck, pred_sck))

    # Conditions
    df_results_complete = df_results[df_results['target_0_1'] > 0]

    # For temperature
    # df_results_complete = df_results_complete[df_results_complete['target_array_real'].between(35.5, 39)]

    # New metrics with real units
    pred_array_real_filt = df_results_complete['pred_array_real'].to_numpy()
    target_array_real_filt = df_results_complete['target_array_real'].to_numpy()

    # Scale to final resolution
    if 'pred_array_offset' in df_results.columns:
        df_results_complete['pred_array_offset'] = df_results['pred_array_offset']
        pred_array_selected = df_results['pred_array_offset'].to_numpy()
    else:
        pred_array_selected = pred_array_real_filt
        df_results_complete['pred_array_offset'] = pred_array_real_filt

    MSE_real = np.divide(np.sum((pred_array_selected - target_array_real_filt) ** 2), len(target_array_real_filt))
    MAE_real = np.divide(np.sum(np.abs(pred_array_selected - target_array_real_filt)), len(target_array_real_filt))
    MAE_std = np.std(np.abs(pred_array_selected - target_array_real_filt))
    print('MSE_real:', np.round(MSE_real, 5))
    print('MAE_real:', np.round(MAE_real, 5))
    print('MAE_real_SD:', np.round(MAE_std, 5))

    pred_sck_r = pred_array_selected.reshape(-1, 1)
    target_sck_r = target_array_real_filt.reshape(-1, 1)
    lin_model_real = LinearRegression()
    lin_model_real.fit(pred_sck_r, target_sck_r)
    print('R2 real: ')
    print(lin_model_real.score(target_sck_r, pred_sck_r))

    # Spearman Correlation
    res = stats.spearmanr(pred_array_selected, target_array_real_filt)
    spearman_corr = res.statistic
    spearman_pval = res.pvalue
    print('spearman_corr: ')
    print(spearman_corr)
    print('spearman_pval: ')
    print(spearman_pval)

# Figures ------------------------------------------------------------------------------------------------------------
# # Figure 1
# fig = plt.figure(figsize=(5, 5))
# plt.xlabel('Target Label')
# plt.ylabel('Predicted Label')
# plt.scatter(target_array, pred_array, marker='o', alpha=0.5, s=10)
# plt.show()
#
# # Figure 2
# fig = plt.figure(figsize=(5, 5))
# plt.xlabel('Target Label')
# plt.ylabel('Predicted Label')
# plt.scatter(target_array_real_filt, pred_array_selected, marker='o', alpha=0.5, s=10)
# plt.show()
#
# plt.plot(target_array, lin_model.predict(target_sck),
#          color='lightgray', alpha=0.5, label='lin reg.')
# plt.plot(target_array, target_array,
#          color='gray', alpha=0.8, label='perfect score')
#
# # Figure 3
# # CONFUSION MATRIX
# indexes = [np.abs(x - real_unique_array).argmin() for x in pred_array_selected]
# predicted_label = [real_unique_array[x] for x in indexes]
# target_label = target_array_real_filt
#
# # Make labels an array or int
# factor = 1
# if TARGET_VAR == 'glucose':
#     factor = 200
# predicted_label_int = [int(x * factor) for x in predicted_label]
# target_label_int = [int(x * factor) for x in target_label]
#
# cm = confusion_matrix(predicted_label_int, target_label_int)
# cm_perc = [x / np.sum(x) if np.sum(x) > 0 else x for x in cm]
#
# # Plot confusion matrix
# sns.heatmap(cm_perc, annot=True, fmt='.1%', cmap='Blues')
# plt.show()

# # Figure 4
# Plot Boxplot

# # For Temperature
# df_results_complete = df_results_complete[df_results_complete['target_array_real'].between(35.5, 39, inclusive='both')]

fig = plt.figure(figsize=(4.5, 4.5))
sns.boxplot(
    data=df_results_complete,
    x="target_array_real",
    y="pred_array_offset",
    # width=1.1
    color='thistle',
)

sns.swarmplot(
    data=df_results_complete,
    x="target_array_real",
    y="pred_array_offset",
    size=2.5,
    color='black',
    alpha=0.6,
    # width=1.1
    # color='thistle',
)

LETTER_SIZE = 13

# --
plt.axvspan(-0.5, 1.5, color='lightgray', alpha=0.4)
plt.xlabel('Sodium Concentration [mg/ml]', fontsize=LETTER_SIZE)
plt.ylabel('Sodium Model Prediction [mg/ml]', fontsize=LETTER_SIZE)

plt.text(-0.2, 10, "MAE=0.848", fontsize=LETTER_SIZE)

plt.xticks([0, 1, 2, 3, 4], [0, 2.0, 4.5, 8.0, 11.5], fontsize=LETTER_SIZE)
plt.yticks(fontsize=LETTER_SIZE)
fig.tight_layout()


SAVE_FIG = True
if SAVE_FIG:
    path_save_fig = config['root_path'] + '/results/unet_mlp_plots_25/'
    plt.savefig(path_save_fig + 'sodium_noise20_2025_08_07.pdf', format='pdf')

plt.show()

# ------------------------

# RESULTS AUG 2025

# Temperature
# offset = -5495
# scale = 149
# plt.axvspan(1.6, 4.4, color='lightgray', alpha=0.4)
# plt.yticks([32, 34, 36, 38, 40, 42], [32.0, 34.0, 36.0, 38.0, 40.0, 42.0])
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], [35.5, 36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0])
# plt.xlabel('Temperature [°C]')
# plt.ylabel('Temperature Model Prediction')
# # orange

# Sodium
# plt.axvspan(-0.5, 1.5, color='lightgray', alpha=0.4)
# plt.xlabel('Sodium Concentration [mg/ml]', fontsize=11)
# plt.ylabel('Sodium Model Prediction [mg/ml]', fontsize=11)
#
# plt.text(-0.2, 10, "MAE=0.857", fontsize=11)
#
# plt.xticks([0, 1, 2, 3, 4], [0, 2.0, 4.5, 8.0, 11.5], fontsize=11)
# plt.yticks(fontsize=11)
# # 'thistle'

# pH
# plt.axvspan(3.5, 4.5, color='lightgray', alpha=0.4)
# plt.xlabel('pH', fontsize=LETTER_SIZE)
# plt.ylabel('pH Model Prediction', fontsize=LETTER_SIZE)
#
# plt.text(-0.2, 8.5, "MAE=0.416", fontsize=LETTER_SIZE)
#
# plt.yticks(
#     [3, 4, 5, 6, 7, 8, 9],
#     [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
#     fontsize=LETTER_SIZE,
# )
#
# plt.xticks(fontsize=LETTER_SIZE)
# # lightskyblue


# Glucose
# plt.axvspan(1.5, 5.4, color='lightgray', alpha=0.4)
#
# plt.xlabel('Glucose Concentration [mg/mL]', fontsize=LETTER_SIZE)
# plt.ylabel('Glucose Model Prediction [mg/mL]', fontsize=LETTER_SIZE)
#
# plt.text(-0.2, 0.55, "MAE=0.022", fontsize=LETTER_SIZE)
# plt.xticks([0, 1, 2, 3, 4, 5],
#            [0.0, 0.04, 0.08, 0.16, 0.31, 0.63],
#            fontsize=LETTER_SIZE
#            )
#
# plt.yticks(fontsize=LETTER_SIZE)
#
# fig.tight_layout()
# # 'lightcoral'
