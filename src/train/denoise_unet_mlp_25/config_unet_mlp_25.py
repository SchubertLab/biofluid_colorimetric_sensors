import numpy as np


# Config
def main():
    ROOT_PATH = '/home/ubuntu/biofluid_colorimetric_sensors'
    SAVE_PATH = ROOT_PATH + '/models/'
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

    # Dataset configs
    N_SYNTHETIC_IMAGES_TRAIN = 240  #240
    N_SYNTHETIC_IMAGES_VAL = 64  # 48
    TARGET_IMAGE_SHAPE = (128, 128)   # (72, 72)  # (32,32) / (128, 128)
    ALLOW_INCOMPLETE = False
    INCOMPLETE_EVERY_N = 10
    ILLUMINATION_NOISE = True
    NOISE_LEVEL = 0.25
    OFFSET = 0.1
    MASKED_OUTPUT = True
    ONE_MASK_INPUT = True

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

    SELECTED_MASK = 'glucose'  # 'all_masks' , 'glucose', 'ph', 'temperature'  #important

    ADDITIONAL_SPECS = {
        'labels_to_sample_all': LABELS_TO_SAMPLE_ALL,
        'list_repeated_areas': LIST_REPEATED_AREAS,
        'selected_mask': SELECTED_MASK,
    }

    AUGMENTATION_SPECS = {
        'image_rotation': [-20, 20],
        'distortion_scale': 0.2,
        'distortion_p': 0.5,
        'resize_crop_down': 0.8,
        'resize_crop_up': 1.3,
    }

    # Model configs
    MODEL_PATH = SAVE_PATH + '/model__unet_masked.pt'
    MODEL_NAME = 'dae_mlp_model'
    MODEL_TYPE = 'AE_LATENT'
    OUTPUT_CHANNELS = 4  # #important
    INPUT_CHANNELS = 4  # #important
    IMAGE_SIZE = TARGET_IMAGE_SHAPE[0]
    SAVE_MODEL = True
    INIT_FILTERS = 32

    DAE_MODEL_CONFIG = {
        'image_size': IMAGE_SIZE,
        'n_in_channels': INPUT_CHANNELS,
        'n_out_channels': OUTPUT_CHANNELS,
        'init_features': INIT_FILTERS,
        'n_conv_blocks': 3,
        'kernel': 2,
        'stride': 2,
        'padding': 0,
    }

    # Training Configs
    OPTIMIZER = 'Adadelta'
    SEED = 0
    BATCH_SIZE = 16
    USE_CUDA = True
    DAE_LOSS_TYPE = 'BCE'  # 'BCE' or 'MSE'
    MLP_LOSS_TYPE = 'L1'  # 'L1' or 'MSE'
    LEARNING_RATE = 0.01
    GAMMA = 0.1
    LR_STEP = 20
    EPOCHS = 3  # 40
    EARLY_STOP = False
    EARLY_STOP_DELTA = 0.001
    PATIENCE = 4
    LOG_INTERVAL = 1

    THRESHOLD_PER_VAR = {
        'glucose': 0.04,
        'ph': 0.17,
        'Na': 0.23,
        'temperature': 0.04,
    }

    ORDER_VARS_PRED = ['glucose', 'ph', 'Na', 'temperature']  # important

    VARIABLE_PRED_MLP_ONLY = 'glucose'

    PROJECT_NAME_WANDB = 'UNET_MLP_Glucose_2025_07'

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
        'illumination_noise': ILLUMINATION_NOISE,
        'noise_level': NOISE_LEVEL,
        'offset': OFFSET,
        'masked_ground_truth': MASKED_OUTPUT,
        'one_mask_input': ONE_MASK_INPUT,

        # model
        'model_path': MODEL_PATH,
        'model_name': MODEL_NAME,
        'model_type': MODEL_TYPE,
        'dae_model_config': DAE_MODEL_CONFIG,

        # training
        'image_size': IMAGE_SIZE,
        'seed': SEED,
        'use_cuda': USE_CUDA,
        'batch_size': BATCH_SIZE,
        'dae_loss_type': DAE_LOSS_TYPE,
        'mlp_loss_type': MLP_LOSS_TYPE,
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
        'order_vars_pred': ORDER_VARS_PRED,
        'threshold_per_var': THRESHOLD_PER_VAR,
        'optimizer': OPTIMIZER,
        'variable_pred_mlp_only': VARIABLE_PRED_MLP_ONLY,
        # WANDB
        'project_name_wandb': PROJECT_NAME_WANDB,
    }
    return CONFIG


if __name__ == '__main__':
    main()
