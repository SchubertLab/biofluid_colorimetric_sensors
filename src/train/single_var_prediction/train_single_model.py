import torch
import torch.optim as optim

from datetime import datetime
from torch.optim.lr_scheduler import StepLR

import src.train.single_var_prediction.config as config_args
import src.models.model_cnn_mlp as model_cnn_mlp
import src.train.single_var_prediction.load_sensor_dataset as sensor_dataset
import src.models.model_resnet_mlp as model_resnet_mlp
import src.train.single_var_prediction.train_test_per_epoch as train_test_per_epoch


def main():
    # Load Configuration
    config = config_args.main()
    torch.manual_seed(config['seed'])

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
    train_dataloader, test_dataloader = sensor_dataset.load_dataset(config)

    # Load Model
    MODEL_TYPE = config['model_type']
    model = None
    if MODEL_TYPE == 'CNN_MLP':
        model_config = config['cnn_model_config']
        model = model_cnn_mlp.CnnMlpNet(
            n_input_channels=model_config['n_input_channels'],
            n_variables_out=model_config['n_variables_out'],
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

    # Optimizer
    optimizer = optim.Adadelta(
        model.parameters(),
        lr=config['lr']
    )

    scheduler = StepLR(
        optimizer,
        step_size=config['lr_step'],
        gamma=config['gamma']
    )

    # Loss function
    if config['loss_type'] == 'L1':
        label_loss = torch.nn.L1Loss(reduction='mean')
    elif config['loss_type'] == 'MSE':
        label_loss = torch.nn.MSELoss(reduction='mean')
    else:
        label_loss = None

    # Array for train/test loss
    train_loss_array = []
    val_loss_array = []

    # date for model saving
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    print("Date and time:", dt_string)

    continue_training = True
    early_stop_counter = 0

    for epoch in range(config['epochs']):
        if continue_training:
            print('Starting epoch ', epoch)
            train_loss_ep, train_mae_per_variable, train_mse_per_variable = train_test_per_epoch.train(
                config=config,
                model=model,
                device=device,
                train_loader=train_dataloader,
                loss=label_loss,
                optimizer=optimizer,
                epoch=epoch,
            )

            val_loss_ep, val_mae_per_var, val_mse_per_var = train_test_per_epoch.test(
                config=config,
                model=model,
                loss=label_loss,
                device=device,
                test_loader=test_dataloader
            )
            scheduler.step()

            train_loss_array.append(train_loss_ep)
            val_loss_array.append(val_loss_ep)

            if len(val_loss_array) > 0:
                # Save model with best loss
                if val_loss_ep <= min(val_loss_array):
                    if config['save_model']:
                        torch.save(
                            model.state_dict(),
                            config['save_path'] + "model_" + config['model_name'] + "_" + dt_string + ".pt"
                        )
                    epoch_best_model = epoch
                    print('Best Model - Epoch nr.', str(epoch_best_model))

                    temp_best_metrics = {
                        'val_loss': val_loss_ep,
                        'best_epoch': epoch_best_model,
                        'val_mae_per_var': val_mae_per_var.cpu().numpy()[0],
                        'val_mse_per_var': val_mse_per_var.cpu().numpy()[0],
                    }

                # Early Stopping
                if config['early_stopping']:
                    if val_loss_ep < min(val_loss_array):
                        early_stop_counter = 0
                    elif val_loss_ep >= min(val_loss_array):
                        early_stop_counter += 1
                        print('early_stop_counter', early_stop_counter)
                        if early_stop_counter >= config['patience']:
                            print('Early Stopping', early_stop_counter)
                            continue_training = False
    print('Finished! :)')
    print('Best Metrics:')
    print(temp_best_metrics)


if __name__ == '__main__':
    main()
