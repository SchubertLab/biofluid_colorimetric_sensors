import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import src.test.single_var_prediction.config_evaluation as config_args
import src.models.model_cnn_mlp as model_cnn_mlp
import src.models.model_resnet_mlp as model_resnet_mlp
import src.train.single_var_prediction.load_sensor_dataset as sensor_dataset


def evaluate(config, model, loss, device, test_loader):
    model.eval()

    # Evaluation metrics
    test_loss = 0
    correct_per_var = torch.from_numpy(np.zeros(len(config['order_vars_pred'])).astype(np.float32)).to(device)
    tensor_mae_per_var = torch.from_numpy(np.zeros(len(config['order_vars_pred'])).astype(np.float32)).to(device)

    # Per variable threshold for classification as correct/incorrect prediction
    threshold_per_var = torch.from_numpy(np.zeros(len(config['order_vars_pred'])).astype(np.float32)).to(device)

    for idx_var, var_pred in enumerate(config['order_vars_pred']):
        threshold_per_var[idx_var] = config['threshold_per_var'][var_pred]

    with torch.no_grad():
        pred_array = []
        target_array = []

        for batch_idx, sample_batch in enumerate(test_loader):
            images_masks_batch = sample_batch['image_masks']
            label_sensor_batch = sample_batch['label_sensor']

            batch_size = images_masks_batch.shape[0]
            target_tensor = torch.from_numpy(
                np.zeros((batch_size, len(config['order_vars_pred'])))
            )

            for idx_var, var_pred in enumerate(config['order_vars_pred']):
                variable_value = label_sensor_batch[var_pred]['norm']
                complete_value = label_sensor_batch[var_pred]['complete']

                # Multiply label by completeness matrix
                temp_tensor = variable_value * complete_value
                target_tensor[:, idx_var] = temp_tensor

            # Model Prediction
            images_masks_batch = images_masks_batch.float()
            target_tensor = target_tensor.float()

            data, target = images_masks_batch.to(device), target_tensor.to(device)
            output = model(data)

            # MSE Loss
            test_loss += loss(output, target).item()

            # Loss for every variable add separately the MAE
            per_batch_loss = torch.nn.L1Loss(reduction='none')
            per_batch_unreduced_loss = per_batch_loss(output, target)
            print('pred', output)
            print('target', target)

            per_var_loss = per_batch_unreduced_loss.to(device)
            mae_per_variable = torch.sum(per_var_loss, dim=0).to(device)
            tensor_mae_per_var += mae_per_variable

            # If prediction is below threshold for each variable then is correct
            for b_var in per_batch_unreduced_loss:
                print('per batch out-pred', b_var)
                below_threshold = b_var < threshold_per_var
                correct_per_var += below_threshold

            if batch_idx % config['log_interval'] == 0:
                print('Test Loss: {:.6f}'.format(
                    test_loss
                ))

            # save outputs to plot
            pred_array.append(output.cpu().detach().numpy())
            target_array.append(target.cpu().detach().numpy())

    size_dataset = len(test_loader.dataset)
    test_loss /= size_dataset

    temp_ones_tensor = torch.from_numpy(np.ones(len(config['order_vars_pred'])).astype(np.float32)).to(device)
    size_dataset_tensor = (size_dataset * temp_ones_tensor).to(device)
    tensor_mae_per_var = torch.div(tensor_mae_per_var, size_dataset_tensor)
    correct_percentage = 100 * torch.div(correct_per_var, size_dataset_tensor)

    print(
        '\nTest set: Average loss: {:.4f}, MAE per var: {}, Correct per Variable: {} ({}%)\n'.format(
            test_loss,
            tensor_mae_per_var,
            correct_per_var,
            correct_percentage
        )
    )
    return pred_array, target_array


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
TARGET_VAR = config['target_var']
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
    _, test_dataloader = sensor_dataset.load_dataset(config)

    # Load Model
    MODEL_TYPE = config['model_type']
    model = None
    if MODEL_TYPE == 'CNN_MLP':
        model_config = config['cnn_model_config']
        model = model_cnn_mlp.CnnMlpNet(
            n_input_channels=model_config['n_input_channels'],
            n_variables_out=model_config['n_variables_out'],
            # n_kernel=model_config['n_kernel'],
            n_filters_c1=model_config['n_filters_c1'],
            n_filters_c2=model_config['n_filters_c2'],
            n_fc1=model_config['n_fc1'],
            n_fc2=model_config['n_fc2'],
            n_flat_input=model_config['n_flat_input'],
            activation_last=model_config['activation_last'],
            dropout_bool=model_config['dropout_bool'],
            dropout_perc=model_config['dropout_perc'],
        )

    elif MODEL_TYPE == 'RESNET_18_MLP':
        model_config = config['resnet_model_config']
        model = model_resnet_mlp.ResnetMLP(
            pretrained=model_config['pretrained'],
            n_input_channels=model_config['n_input_channels'],
            n_variables_out=model_config['n_variables_out'],
            n_flat_input=model_config['n_flat_input'],
            activation_last=model_config['activation_last'],
        )

        model = model.to(device)
        print(model)

    # Load model Pretrained weights
    model_path = config['model_path']

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint)

    # Loss function
    if config['loss_type'] == 'L1':
        label_loss = torch.nn.L1Loss(reduction='mean')
    elif config['loss_type'] == 'MSE':
        label_loss = torch.nn.MSELoss(reduction='mean')
    else:
        label_loss = None

    pred_array, target_array = evaluate(
        config=config,
        model=model,
        loss=label_loss,
        device=device,
        test_loader=test_dataloader
    )

    pred_array = np.array(np.concatenate(pred_array).ravel())
    target_array = np.array(np.concatenate(target_array).ravel())

    # Real units for predictions ------------------------------------------------------------------------------------
    # Slope and intersect to translate units

    if config['allow_incomplete']:
        unique_norm = np.unique(target_array)[1:]
    else:
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
    if config['save_results']:
        df_results.to_csv(
            config['model_results_path'],
            index=False
        )

else:
    df_results = pd.read_csv(config['model_results_path'])
    target_array = df_results['target_0_1'].to_numpy()
    pred_array = df_results['pred_0_1'].to_numpy()

# Metrics with 0-1 scale
# Fit Linear model with zero
pred_sck = pred_array.reshape(-1, 1)
target_sck = target_array.reshape(-1, 1)
lin_model = LinearRegression()
lin_model.fit(pred_sck, target_sck)
print('R_norm: ')
print(lin_model.score(target_sck, pred_sck))

# Find MAE and MSE
MSE_final = np.divide(np.sum((pred_array - target_array) ** 2), len(target_array))
MAE_final = np.divide(np.sum(np.abs(pred_array - target_array)), len(target_array))
print('MSE_norm:', MSE_final)
print('MAE_norm:', MAE_final)

df_results_complete = df_results[df_results['target_0_1'] > 0]

# For Temperature
# df_results_complete = df_results_complete[
#     df_results_complete['target_array_real'].between(35.8, 39.0)   # 35.5, 39
# ]

# New metrics with real units
pred_array_real_filt = df_results_complete['pred_array_real'].to_numpy()
target_array_real_filt = df_results_complete['target_array_real'].to_numpy()

# if offset and factor
APPLY_OFFSET = False
if APPLY_OFFSET:
    offset = 0
    scale = 1
    pred_array_offset = (pred_array_real_filt - offset) * scale
    df_results_complete['pred_array_offset'] = pred_array_offset
else:
    pred_array_offset = df_results_complete['pred_array_offset']

# choose what to plot -> pred_array_real_filt, pred_array_offset
pred_array_selected = pred_array_offset.to_numpy()

MSE_real = np.divide(np.sum((pred_array_selected - target_array_real_filt) ** 2), len(target_array_real_filt))
MAE_real = np.divide(np.sum(np.abs(pred_array_selected - target_array_real_filt)), len(target_array_real_filt))
MAE_std = np.std(np.abs(pred_array_selected - target_array_real_filt))
print('MSE_real:', MSE_real)
print('MAE_real:', MAE_real)
print('MAE_real_SD:', MAE_std)

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


# --------------------------------------------------------------------------------------------------------
# # Figure 1
# fig = plt.figure(figsize=(5, 5))
# plt.xlabel('Target Label')
# plt.ylabel('Predicted Label')
# plt.scatter(target_array, pred_array, marker='o', alpha=0.5, s=10)
# plt.show()

# # Figure 2
# fig = plt.figure(figsize=(5, 5))
# plt.xlabel('Target Label')
# plt.ylabel('Predicted Label')
# plt.scatter(target_array_real_filt, pred_array_selected, marker='o', alpha=0.5, s=10)
# plt.show()


# plt.plot(target_array, lin_model.predict(target_sck),
#             color='lightgray', alpha=0.5, label='lin reg.')
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
#
# predicted_label_int = [int(x * factor) for x in predicted_label]
# target_label_int = [int(x * factor) for x in target_label]
#
# cm = confusion_matrix(predicted_label_int, target_label_int)
# cm_perc = [x / np.sum(x) if np.sum(x) > 0 else x for x in cm]
#
# # Plot confusion matrix
# sns.heatmap(cm_perc, annot=True, fmt='.1%', cmap='Blues')
# plt.show()

# Plot violin
fig = plt.figure(figsize=(3.9, 4))   # (3.5, 4) or (3,4) for normal
sns.boxplot(
    data=df_results_complete,
    x="target_array_real",
    y="pred_array_offset",
    # width=1.1
    color='orange',
)
sns.swarmplot(
    data=df_results_complete,
    x="target_array_real",
    y="pred_array_offset",
    size=2,
    color='black',
    alpha=0.6,
    # width=1.1
    # color='thistle',
)

plt.axvspan(1.6, 4.4, color='lightgray', alpha=0.4)
plt.xlabel('Temperature [°C]')
plt.ylabel('Temperature Model Prediction')
# plt.yticks([35,36,37,38,39,40], [35.0,36.0,37.0,38.0,39.0,40.0])


fig.tight_layout()

# Temperature
# plt.axvspan(1.6, 4.4, color='lightgray', alpha=0.4)
# plt.xlabel('Temperature [°C]')
# plt.ylabel('Temperature Model Prediction')
# plt.yticks([35,36,37,38,39,40], [35.0,36.0,37.0,38.0,39.0,40.0])
# # 'orange'
# # figsize=(3.9, 4)

# Sodium
# plt.axvspan(-0.4, 1.4, color='lightgray', alpha=0.4)
# plt.xlabel('Sodium Concentration [mg/ml]')
# plt.ylabel('Sodium Model Prediction')
# plt.xticks([0, 1, 2, 3, 4], [0, 2.0, 4.5, 8.0, 11.5])
# plt.ylim([-2.4, 17])
# 'thistle'

# # pH
# plt.axvspan(3.6, 4.4, color='lightgray', alpha=0.4)
# plt.xlabel('pH')
# plt.ylabel('pH Model Prediction')
# plt.yticks([3,4,5,6,7,8,9], [3.0,4.0,5.0,6.0,7.0,8.0,9.0])
# # figsize=(3.5, 4)
# # lightskyblue


# # Glucose
# plt.axvspan(1.6, 5.4, color='lightgray', alpha=0.4)
# plt.xlabel('Glucose Concentration [mg/mL]')
# plt.ylabel('Glucose Model Prediction')
# plt.xticks([0, 1, 2, 3, 4, 5, 6], [3,4,5,6,7,8,9])
# plt.yticks([3,4,5,6,7,8,9], [3,4,5,6,7,8,9])
# plt.xticks([0, 1, 2, 3, 4], [0, 2.0, 4.5, 8.0, 11.5])
# plt.ylim([-2.45, 17])
# plt.xticks([0, 1, 2, 3, 4, 5], [0.0, 0.04, 0.08, 0.16, 0.31, 0.63])
# 'lightcoral'


SAVE_FIG = False
if SAVE_FIG:
    temp_var = 'temperature'
    path_save_fig = config['root_path'] + '/results/single_var_models/' + temp_var + '/'
    plt.savefig(path_save_fig + temp_var + '_predictions_2025_09_15_baseline_pnas.pdf', format='pdf')
plt.show()

# Evaluate Thresholds
EVALUATE_THRESHOLD = False

if EVALUATE_THRESHOLD:

    # # Temperature
    # THRESHOLD_RANGE = [36.5, 37.5]

    # # pH
    # THRESHOLD_RANGE = [7, 7.9]
    #
    # # Sodium
    # THRESHOLD_RANGE = [3.6, 50]  # Hypernatremia
    # THRESHOLD_RANGE = [2.0, 50]  # Cystic Fibrosis
    #
    # # Glucose
    THRESHOLD_RANGE = [-25, 0.5]  # Hypoglycemia

    target_boolean_threshold = df_results_complete['target_array_real'].between(THRESHOLD_RANGE[0], THRESHOLD_RANGE[1])
    pred_boolean_threshold = df_results_complete['pred_array_offset'].between(THRESHOLD_RANGE[0], THRESHOLD_RANGE[1])

    accuracy_result = balanced_accuracy_score(target_boolean_threshold, pred_boolean_threshold)
    f1_result = f1_score(target_boolean_threshold, pred_boolean_threshold, average='weighted')
    print('accuracy_result', accuracy_result)
    print('f1_result', f1_result)

# test
# ROOT_PATH = config['root_path']
#
#
# MODEL_RESULTS_PATH = (ROOT_PATH + 'results/single_var_models/temperature/'
#                       + 'predictions_2025-04-07_model_TEMPERATURE_clean.csv')
#
# df_results_save = df_results_complete[[
#     'target_0_1',
#     'pred_0_1',
#     'target_array_real',
#     'pred_array_offset'
# ]]
#
# df_results_save.to_csv(
#     MODEL_RESULTS_PATH,
#     index=False
# )
