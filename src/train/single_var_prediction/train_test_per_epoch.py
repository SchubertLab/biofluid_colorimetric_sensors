import torch
import numpy as np


# Train function
def train(config, model, device, train_loader, loss, optimizer, epoch):

    train_loss = None
    mae_per_variable = None
    mse_per_variable = None

    model.train()
    for batch_idx, sample_batch in enumerate(train_loader):
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

        images_masks_batch = images_masks_batch.float()
        target_tensor = target_tensor.float()

        data, target = images_masks_batch.to(device), target_tensor.to(device)
        optimizer.zero_grad()

        # Model Prediction
        output = model(data)

        # Loss for all variables
        train_loss = loss(output, target)

        # Metrics for the last training batch
        mae_fx = torch.nn.L1Loss(reduction='none')
        tensor_mae_per_batch = mae_fx(output, target)
        mae_per_variable = torch.divide(
            torch.sum(tensor_mae_per_batch, dim=0).to(device),
            batch_size
        )

        mse_fx = torch.nn.MSELoss(reduction='none')
        tensor_mse_per_batch = mse_fx(output, target)
        mse_per_variable = torch.divide(
            torch.sum(tensor_mse_per_batch, dim=0).to(device),
            batch_size
        )

        # Optimize
        train_loss.backward()
        optimizer.step()
        if batch_idx % config['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), train_loss.item()))
            if config['dry_run']:
                break

    return train_loss, mae_per_variable, mse_per_variable


def test(config, model, loss, device, test_loader):
    model.eval()

    # Evaluation metrics
    test_loss = 0
    correct_per_var = torch.from_numpy(np.zeros(len(config['order_vars_pred'])).astype(np.float32)).to(device)
    tensor_mae_per_var = torch.from_numpy(np.zeros(len(config['order_vars_pred'])).astype(np.float32)).to(device)
    tensor_mse_per_var = torch.from_numpy(np.zeros(len(config['order_vars_pred'])).astype(np.float32)).to(device)

    # Per variable threshold for classification as correct/incorrect prediction
    threshold_per_var = torch.from_numpy(np.zeros(len(config['order_vars_pred'])).astype(np.float32)).to(device)

    for idx_var, var_pred in enumerate(config['order_vars_pred']):
        threshold_per_var[idx_var] = config['threshold_per_var'][var_pred]

    with torch.no_grad():
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

            # Loss
            test_loss += loss(output, target).item()

            # Metrics
            # MAE for every variable
            mae_loss_fx = torch.nn.L1Loss(reduction='none')
            per_batch_mae_loss = mae_loss_fx(output, target).to(device)
            mae_per_variable = torch.sum(per_batch_mae_loss, dim=0).to(device)
            tensor_mae_per_var += mae_per_variable

            # MSE for every variable
            mse_loss_fx = torch.nn.MSELoss(reduction='none')
            per_batch_mse_loss = mse_loss_fx(output, target).to(device)
            mse_per_variable = torch.sum(per_batch_mse_loss, dim=0).to(device)
            tensor_mse_per_var += mse_per_variable

            # # If prediction is below threshold for each variable then is correct
            # for b_var in per_batch_unreduced_loss:
            #     below_threshold = b_var < threshold_per_var
            #     correct_per_var += below_threshold
            #
            # if batch_idx % config['log_interval'] == 0:
            #     print('Test Loss: {:.6f}'.format(
            #         test_loss
            #     ))

    size_dataset = len(test_loader.dataset)
    test_loss /= size_dataset

    temp_ones_tensor = torch.from_numpy(np.ones(len(config['order_vars_pred'])).astype(np.float32)).to(device)
    size_dataset_tensor = (size_dataset * temp_ones_tensor).to(device)

    # Return the average MSE and MAE for all the predictions in all the batches
    tensor_mae_per_var = torch.div(tensor_mae_per_var, size_dataset_tensor)
    tensor_mse_per_var = torch.div(tensor_mse_per_var, size_dataset_tensor)

    # r_squared_value
    # correct_percentage = 100 * torch.div(correct_per_var, size_dataset_tensor)

    print(
        '\nTest set: Average loss: {:.4f}, MAE per var: {}, MSE per Variable: {}\n'.format(
            test_loss,
            tensor_mae_per_var,
            tensor_mse_per_var,
            # correct_per_var,
            # correct_percentage
        )
    )
    return test_loss, tensor_mae_per_var, tensor_mse_per_var
