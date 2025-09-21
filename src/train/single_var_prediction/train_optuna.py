import wandb
import torch
import optuna
from datetime import datetime

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import src.train.single_var_prediction.config as config_args
import src.models.model_cnn_mlp as model_cnn_mlp
import src.train.single_var_prediction.load_sensor_dataset as sensor_dataset
import src.models.model_resnet_mlp as model_resnet_mlp
import src.train.single_var_prediction.train_test_per_epoch as train_test_per_epoch


def objective(trial, params):
    # Optuna and WandB configuration ----------------------------------------------------------------------------
    # Cuda Device
    if params['use_cuda'] and torch.cuda.is_available():
        print('cuda available:', torch.cuda.is_available())
        device = torch.device("cuda")
        torch.backends.cudnn.enabled = False  # (needed for Denbi?)
    else:
        device = torch.device("cpu")
    print('device:', device)

    # Parameters to be optimized with the trial
    # Model Configurations
    MODEL_TYPE = trial.suggest_categorical(
        'model_type',
        ['CNN_MLP', 'RESNET_18_MLP']
    )

    LAST_NEURON_ACTIVATION = trial.suggest_categorical(
        'last_neuron_activation',
        ['sigmoid', 'softplus']
    )

    if MODEL_TYPE == 'CNN_MLP':
        # CNN_DROPOUT = trial.suggest_categorical(
        #     'cnn_dropout',
        #     [True, False]
        # )
        model_config = params['cnn_model_config']
        model_config['activation_last'] = LAST_NEURON_ACTIVATION
        # model_config['dropout_bool'] = CNN_DROPOUT

    elif MODEL_TYPE == 'RESNET_18_MLP':
        RESNET_PRETRAINED = False
        # RESNET_PRETRAINED = trial.suggest_categorical(
        #     'resnet_pretrained',
        #     [True, False]
        # )
        model_config = params['resnet_model_config']
        model_config['activation_last'] = LAST_NEURON_ACTIVATION
        model_config['pretrained'] = RESNET_PRETRAINED

    # Training
    LEARNING_RATE = trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True)

    # Optimization parameters
    params['MODEL_TYPE'] = MODEL_TYPE
    params['ACTIVATION_LAST'] = LAST_NEURON_ACTIVATION
    params['lr'] = LEARNING_RATE

    # WandB configuration
    run = wandb.init(
        project=PROJECT_NAME_WANDB,
        config=params,
    )
    config_wandb = wandb.config
    print('wandb', str(run.id), str(run.name))

    # Dataset, Model and Optimizer ----------------------------------------------------------------------------

    # Load Dataset
    train_dataloader, test_dataloader = sensor_dataset.load_dataset(params)

    # Model
    model = None
    if MODEL_TYPE == 'CNN_MLP':
        model = model_cnn_mlp.CnnMlpNet(
            n_input_channels=model_config['n_input_channels'],
            n_variables_out=model_config['n_variables_out'],
            n_kernel=model_config['n_kernel'],
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
        model = model_resnet_mlp.ResnetMLP(
            pretrained=model_config['pretrained'],
            n_input_channels=model_config['n_input_channels'],
            n_variables_out=model_config['n_variables_out'],
            n_flat_input=model_config['n_flat_input'],
            activation_last=model_config['activation_last'],
        )

    model = model.to(device)
    print(model)

    # Optimizer
    optimizer = optim.Adadelta(
        model.parameters(),
        lr=params['lr']
    )

    scheduler = StepLR(
        optimizer,
        step_size=params['lr_step'],
        gamma=params['gamma'],
    )

    # Loss function
    if params['loss_type'] == 'L1':
        label_loss = torch.nn.L1Loss(reduction='mean')
    elif params['loss_type'] == 'MSE':
        label_loss = torch.nn.MSELoss(reduction='mean')
    else:
        label_loss = None

    # Parallelize
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # Train and Validation   ------------------------------------------------------------------------------------------

    # Date for model saving
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

    # Initialize Metrics per Epoch
    train_loss_per_epoch = []
    val_loss_per_epoch = []

    metrics_per_epoch = {
        'mae_per_variable': [],
        'mse_per_variable': [],
    }
    train_metrics_per_epoch = metrics_per_epoch.copy()
    val_metrics_per_epoch = metrics_per_epoch.copy()

    # Log Model Training to WandB
    wandb.watch(model, log_freq=100)
    continue_training = True
    early_stop_counter = 0

    for epoch in range(params['epochs']):
        if continue_training:
            # Model Training ------------------------------------------------------------------------------------
            print('Starting epoch ', epoch)
            train_loss_ep, train_mae_per_variable, train_mse_per_variable = train_test_per_epoch.train(
                config=params,
                model=model,
                device=device,
                train_loader=train_dataloader,
                loss=label_loss,
                optimizer=optimizer,
                epoch=epoch,
            )
            # save train metrics
            train_loss_per_epoch.append(train_loss_ep)
            train_mae_per_variable_float = train_mae_per_variable.detach().cpu().numpy()[0]  # todo-> 0 to variable name
            train_mse_per_variable_float = train_mse_per_variable.detach().cpu().numpy()[0]
            train_metrics_per_epoch['mae_per_variable'].append(train_mae_per_variable_float)
            train_metrics_per_epoch['mse_per_variable'].append(train_mse_per_variable_float)

            # Log training results to Wandb
            wandb.log({"epoch": epoch})
            wandb.log({"train_loss": train_loss_ep})
            wandb.log({"train_mae": train_mae_per_variable_float})
            wandb.log({"train_rmse": train_mse_per_variable_float})

            # Validate the Model  ---------------------------------------------------------------------------------
            val_loss_ep, val_mae_per_var, val_mse_per_var = train_test_per_epoch.test(
                config=params,
                model=model,
                loss=label_loss,
                device=device,
                test_loader=test_dataloader  # TODO: change to val_dataloader
            )
            scheduler.step()

            # save validation metrics
            val_mae_per_var_float = val_mae_per_var.cpu().numpy()[0]  # todo-> 0 to variable name
            val_mse_per_var_float = val_mse_per_var.cpu().numpy()[0]
            val_metrics_per_epoch['mae_per_variable'].append(val_mae_per_var_float)
            val_metrics_per_epoch['mse_per_variable'].append(val_mse_per_var_float)

            # Log validation results to Wandb
            wandb.log({"val_loss": val_loss_ep})
            wandb.log({"val_mae": val_mae_per_var_float})
            wandb.log({"val_rmse": val_mse_per_var_float})
            # wandb.log({"val_pearson_r": val_mse_per_var})

            trial.report(val_mae_per_var_float, epoch)
            trial.report(val_mse_per_var_float, epoch)

            if len(val_loss_per_epoch) > 0:
                # Save model with best loss
                if val_loss_ep <= min(val_loss_per_epoch):
                    if params['save_model']:
                        torch.save(
                            model.state_dict(),
                            params['save_path'] + "model_" + dt_string + ".pt"
                        )
                    epoch_best_model = epoch
                    print('Best Model - Epoch nr.', str(epoch_best_model))

                    temp_best_metrics = {
                        'val_loss': val_loss_ep,
                        'best_epoch': epoch_best_model,
                        'mae': min(val_metrics_per_epoch['mae_per_variable']),
                        'mse': min(val_metrics_per_epoch['mse_per_variable']),
                    }
                    print(temp_best_metrics)

                # Early Stopping
                if params['early_stopping']:
                    if val_loss_ep < min(val_loss_per_epoch):
                        early_stop_counter = 0
                    elif val_loss_ep >= min(val_loss_per_epoch):
                        early_stop_counter += 1
                        if early_stop_counter >= params['patience']:
                            continue_training = False
            else:
                temp_best_metrics = {
                    'val_loss': val_loss_ep,
                    'best_epoch': 0,
                    'mae': val_mae_per_var_float,
                    'mse': val_mse_per_var_float,
                }
            val_loss_per_epoch.append(val_loss_ep)

    wandb.finish()

    # Report best score to Optuna
    print('Optuna - Validation Best Metrics')
    print('MAE', temp_best_metrics['mae'])
    print('MSE', temp_best_metrics['mse'])
    # print('PEARSON_R', temp_best_metrics['PEARSON_R'])

    metric_name = params['optuna_metric']
    score_optuna = temp_best_metrics[metric_name]
    return score_optuna


if __name__ == "__main__":
    torch.cuda.manual_seed_all(0)

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    print("Date and time:", dt_string)

    # Configuration File
    config_parameters = config_args.main()

    SAVE_MODEL = False
    OPTUNA_N_TRIALS = config_parameters['optuna_n_trials']
    OPTUNA_STUDY_NAME = config_parameters['project_name_optuna']

    PROJECT_NAME_WANDB = config_parameters['project_name_wandb']

    # ------------------------------------------------------------------------------------------
    # Create optuna study
    storage_name = "sqlite:///{}.db".format(OPTUNA_STUDY_NAME)
    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
    )
    print(f"Sampler is {study.sampler.__class__.__name__}")
    study.optimize(
        lambda trial: objective(
            trial,
            params=config_parameters,
        ),
        n_trials=OPTUNA_N_TRIALS
    )

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    df_optimization = study.trials_dataframe()
    df_optimization.to_csv(config_parameters['save_path_optuna'] + OPTUNA_STUDY_NAME + '.csv')
    SAVE_PATH_OPTUNA = config_parameters['root_path'] + '/data/optuna_trials/'
    df_optimization.to_csv(SAVE_PATH_OPTUNA + OPTUNA_STUDY_NAME + '.csv')
