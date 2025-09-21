import gc
import torch
import wandb
import numpy as np
import torch.optim as optim

from datetime import datetime
from torch.optim.lr_scheduler import StepLR

import src.train.denoise_unet_mlp_25.config_unet_mlp_25 as config_args
import src.models.model_denoising_unet_mlp_25 as model_denoising_mlp
import src.train.single_var_prediction.load_noise_dataset as sensor_noise_dataset
import src.train.denoise_unet_mlp_25.train_test_unet_mlp_per_epoch as train_test_denoiser_per_epoch


def main():
    # Load Configuration
    config = config_args.main()
    torch.manual_seed(config['seed'])

    # WandB
    # WandB configuration
    run = wandb.init(
        project=config['project_name_wandb'],
        config=config,
    )
    print('wandb', str(run.id), str(run.name))

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
    train_dataloader, val_dataloader = sensor_noise_dataset.load_noise_dataset(config)

    # Load Model
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

    # Optimizer
    dae_optimizer = optim.Adadelta(
        dae_mlp_model.parameters(),
        lr=config['lr']
    )

    scheduler = StepLR(
        dae_optimizer,
        step_size=config['lr_step'],
        gamma=config['gamma']
    )

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

    # MLP Loss
    if config['mlp_loss_type'] == 'L1':
        mlp_loss = torch.nn.L1Loss(reduction='mean')
    elif config['mlp_loss_type'] == 'MSE':
        mlp_loss = torch.nn.MSELoss(reduction='mean')
    else:
        mlp_loss = None

    # Array for train/test loss
    train_loss_array = []
    val_loss_array = []

    # date for model saving
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    print("Date and time:", dt_string)

    continue_training = True
    early_stop_counter = 0
    temp_best_metrics = None

    # # Log Model Training to WandB
    wandb.watch(dae_mlp_model, log_freq=100)

    for epoch in range(config['epochs']):
        if continue_training:
            print('Starting epoch ', epoch)

            # Train denoising only in the first 4 epochs
            if epoch < 5:
                multitask_weights = [0.3, 0.3, 0.3, 0.3, 0]
            else:
                multitask_weights = [0.07, 0.07, 0.07, 0.07, 2.6]

            train_multitask_loss, train_denoising_loss,  train_mlp_loss = train_test_denoiser_per_epoch.train(
                config=config,
                denoising_mlp_model=dae_mlp_model,
                device=device,
                train_loader=train_dataloader,
                denoising_loss_dict=dae_loss_dict,
                mlp_loss=mlp_loss,
                optimizer=dae_optimizer,
                epoch=epoch,
                weights=multitask_weights,
            )

            train_loss_float = train_multitask_loss.detach().cpu().numpy()
            train_loss_array.append(train_loss_float)

            # Log training results to Wandb
            wandb.log({"epoch": epoch})
            wandb.log({"train_loss": train_loss_float})
            wandb.log({"train_loss_denoising": train_denoising_loss.detach().cpu().numpy()})
            wandb.log({"train_loss_mlp": train_mlp_loss.detach().cpu().numpy()})

            # Test
            val_multitask_loss, val_denoising_loss,  val_mlp_loss = train_test_denoiser_per_epoch.test(
                config=config,
                denoising_mlp_model=dae_mlp_model,
                device=device,
                val_loader=val_dataloader,
                denoising_loss_dict=dae_loss_dict,
                mlp_loss=mlp_loss,
                epoch=epoch,
                weights=multitask_weights,
            )

            scheduler.step()

            val_multitask_loss_float = val_multitask_loss.cpu().numpy()
            val_loss_array.append(val_multitask_loss_float)

            # Log training results to Wandb
            wandb.log({"val_loss": val_multitask_loss_float})
            wandb.log({"val_loss_denoising": val_denoising_loss.cpu().numpy()})
            wandb.log({"val_mlp_loss": val_mlp_loss.cpu().numpy()})

            # Save Model with best lost
            if len(val_loss_array) > 0:
                # Save model with the best loss in the last stage
                if val_multitask_loss_float <= min(np.array(val_loss_array)):
                    if config['save_model']:
                        torch.save(
                            dae_mlp_model.state_dict(),
                            config['save_path'] + "model_" + config['model_name'] + "_" + dt_string + ".pt"
                        )
                    epoch_best_model = epoch
                    print('Best Model - Epoch nr.', str(epoch_best_model))

                    temp_best_metrics = {
                        'val_loss': val_multitask_loss.cpu().numpy(),
                        'best_epoch': epoch_best_model,
                        'val_loss_denoising': val_denoising_loss.cpu().numpy(),
                    }

                # Early Stopping
                if config['early_stopping']:
                    if val_multitask_loss < min(val_loss_array):
                        early_stop_counter = 0
                    elif val_multitask_loss >= min(val_loss_array):
                        early_stop_counter += 1
                        print('early_stop_counter', early_stop_counter)
                        if early_stop_counter >= config['patience']:
                            print('Early Stopping', early_stop_counter)
                            continue_training = False

    wandb.finish()
    gc.collect()
    torch.cuda.empty_cache()
    print('Finished! :)')
    print('Best Metrics:')
    print(temp_best_metrics)


if __name__ == '__main__':
    main()
