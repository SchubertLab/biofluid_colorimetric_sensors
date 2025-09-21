import numpy as np


# Config
def main():
    ROOT_PATH = '/Users/castelblanco/Documents/PhD-Repos/neonatal_sensors/multiarray_color_sensors/'
    SAVE_PATH = ROOT_PATH + '/models/'
    PATH_IMAGE_SKELETONS = 'data/processed/2024_sensor_skeletons/'
    PATHS_LIST_REF_COLOR_DICT = {
        'glucose': 'data/processed/2024_glucose_ref/glucose_ref_pixels_2025_01_29.pkl',
        'ph': 'data/processed/2024_ph_ref/ph_ref_pixels_2024_07_30.pkl',
        'Na': 'data/processed/2024_na_ref/sodium_ref_pixels_2025_02_05.pkl',
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
            'real': np.array([0, 2.0, 4.5, 8.0, 11.5]),
        },
        'temperature': {
            'real': np.arange(34, 40, 0.5),
        },
    }

    N_SYNTHETIC_IMAGES_TRAIN = 256  # 64 TODO: change to real dataset size
    N_SYNTHETIC_IMAGES_VAL = 500  # 256
    TARGET_IMAGE_SHAPE = (128, 128)
    ALLOW_INCOMPLETE = False
    INCOMPLETE_EVERY_N = 10

    # additional specs
    LABELS_TO_SAMPLE_ALL = {
        'glucose': ['g_ul', 'g_ur', 'g_d'],
        'ph': ['ph_ul', 'ph_ur', 'ph_d'],
        'Na': ['Na'],
        'temperature': ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
    }

    LIST_REPEATED_AREAS = {
        'glucose': ['g_ul', 'g_ur', 'g_d'],
        'ph': None,
        'Na': None,
        'temperature': None,
    }

    SELECTED_MASK = 'temperature'  # 'all' , 'glucose', 'ph', 'temperature'  #important

    ADDITIONAL_SPECS = {
        'labels_to_sample_all': LABELS_TO_SAMPLE_ALL,
        'list_repeated_areas': LIST_REPEATED_AREAS,
        'selected_mask': SELECTED_MASK,
    }

    AUGMENTATION_SPECS = {
        'image_rotation': [-20, 20],
        'distortion_scale': 0.2,
        'distortion_p': 0.5,
    }

    # Model configs
    TARGET_VAR = 'temperature'
    MODEL_NAME = 'model_TEMP_silvery-wildflower'
    MODEL_PATH = '../../../models/single_var/temperature/model_TEMP_silvery-wildflower-16_complete_256_2025_01_29_10_39_58.pt'
    # MODEL_RESULTS_PATH = ROOT_PATH + 'results/single_var_models/temperature/predictions_' + MODEL_NAME + '.csv'

    MODEL_RESULTS_PATH = (ROOT_PATH + 'results/single_var_models/temperature/'
                          + 'predictions_2025-09-15_model_TEMP_silvery-wildflower_pnas.csv')


    # Best Results
    # sodium = model_SODIUM_cnn-mlp-sigm-2_2025_02_05_11_00_31.pt
    # glucose = model_GLUCOSE_valiant-puddle-12_complete256_2025_01_29_13_46_59.pt
    # ph = sensor_cnn_mlp_2024_11_06_16_45_07.pt
    # temperature = sensor_cnn_mlp_2024_12_04_11_00_36_epoch_13.pt
    # 2025- Temperature - silvery-wildflower.pt

    # Best Models
    # glucose = predictions_2025-04-07_model_GLUCOSE_valiant-puddle.csv
    # ph = predictions_2025-04-07_PH_azure-paper-14_2025_01_21.csv
    # sodium = predictions_2025-04-07_model_SODIUM_cnn-mlp-sigm-2_2025_02_05.csv
    # temperature = predictions_2025-04-07_model_TEMP_silvery-wildflower.csv

    MODEL_TYPE = 'RESNET_18_MLP'  # 'CNN_MLP' or 'RESNET_18_MLP'
    N_VARIABLES_OUT = 1  # #important
    INPUT_CHANNELS = 4  # #important
    IMAGE_SIZE = 128
    SAVE_MODEL = False
    SAVE_RESULTS = True

    CNN_MODEL_CONFIG = {
        'n_input_channels': INPUT_CHANNELS,
        'n_variables_out': N_VARIABLES_OUT,
        'n_kernel': 5,
        'n_filters_c1': 32,
        'n_filters_c2': 64,
        'n_fc1': 2600,
        'n_fc2': 1200,
        'activation_last': 'sigmoid',  # sigmoid, relu, silu, softplus
        'dropout_bool': False,
        'dropout_perc': 0.05,
        'n_flat_input': 53824,
    }

    RESNET_MODEL_CONFIG = {
        'pretrained': False,
        'n_input_channels': INPUT_CHANNELS,
        'n_variables_out': N_VARIABLES_OUT,
        'n_flat_input': 1000,
        'activation_last': 'sigmoid',
    }

    # Training Configs
    OPTIMIZER = 'Adadelta'
    SEED = 0
    BATCH_SIZE = 16
    USE_CUDA = True
    LOSS_TYPE = 'L1'  # or 'MSE'
    LEARNING_RATE = 0.005
    GAMMA = 0.1
    LR_STEP = 10
    EPOCHS = 20  # 30
    EARLY_STOP = True
    EARLY_STOP_DELTA = 0.001
    PATIENCE = 3
    DRY_RUN = False
    LOG_INTERVAL = 1
    ALPHA_LOSS = {
        'glucose': 0.25,
        'ph': 0.25,
        'Na': 0.25,
        'temperature': 0.25,
    }
    THRESHOLD_PER_VAR = {
        'glucose': 0.04,
        'ph': 0.17,
        'Na': 0.23,
        'temperature': 0.04,
    }

    ORDER_VARS_PRED = ['temperature']  # ['glucose', 'ph', 'Na', 'temperature']  #important

    # Config Dictionary
    CONFIG = {
        # dataset
        'root_path': ROOT_PATH,
        'path_image_skeletons': PATH_IMAGE_SKELETONS,
        'paths_list_ref_color_dict': PATHS_LIST_REF_COLOR_DICT,
        'label_dictionary': LABEL_DICTIONARY,
        'n_synthetic_images_train': N_SYNTHETIC_IMAGES_TRAIN,
        'n_synthetic_images_val': N_SYNTHETIC_IMAGES_VAL,
        'target_image_shape': TARGET_IMAGE_SHAPE,
        'additional_specs': ADDITIONAL_SPECS,
        'augmentation_specs': AUGMENTATION_SPECS,
        'allow_incomplete': ALLOW_INCOMPLETE,
        'incomplete_every_n': INCOMPLETE_EVERY_N,

        # model
        'target_var': TARGET_VAR,
        'model_path': MODEL_PATH,
        'model_name': MODEL_NAME,
        'model_results_path': MODEL_RESULTS_PATH,
        'model_type': MODEL_TYPE,
        'cnn_model_config': CNN_MODEL_CONFIG,
        'resnet_model_config': RESNET_MODEL_CONFIG,
        'save_results': SAVE_RESULTS,

        # training
        'image_size': IMAGE_SIZE,
        'seed': SEED,
        'use_cuda': USE_CUDA,
        'batch_size': BATCH_SIZE,
        'loss_type': LOSS_TYPE,
        'lr': LEARNING_RATE,
        'gamma': GAMMA,
        'lr_step': LR_STEP,
        'epochs': EPOCHS,
        'save_model': SAVE_MODEL,
        'save_path': SAVE_PATH,
        'early_stopping': EARLY_STOP,
        'early_stop_delta': EARLY_STOP_DELTA,
        'patience': PATIENCE,
        'log_interval': LOG_INTERVAL,
        'dry_run': DRY_RUN,
        'alpha_loss': ALPHA_LOSS,
        'order_vars_pred': ORDER_VARS_PRED,
        'threshold_per_var': THRESHOLD_PER_VAR,
        'optimizer': OPTIMIZER,
    }
    return CONFIG


if __name__ == '__main__':
    main()
    # TODO: convert to args parse.
