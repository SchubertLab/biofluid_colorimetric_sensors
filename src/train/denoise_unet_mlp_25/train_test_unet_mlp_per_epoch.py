import torch
# import time


# Train Function
def train(config, denoising_mlp_model, device, train_loader,
          denoising_loss_dict, mlp_loss, optimizer, epoch,
          weights=None,
          ):

    if weights is None:
        weights = [0.07, 0.07, 0.07, 0.07, 2.6]

    denoising_mlp_model.train()
    total_loss_weighted, denoising_loss_train, mlp_loss_train = None, None, None

    # Train per batch
    for batch_idx, sample_batch in enumerate(train_loader):

        # Sampled images
        noisy_image_plus_masks_batch = sample_batch['image_and_masks']
        original_image_batch = sample_batch['original_image']
        light_kernel_batch = sample_batch['light_kernel']

        # Sampled labels
        labels_batch = sample_batch['label_sensor']
        label_noise_batch = sample_batch['label_noise']

        # Data to device
        input_data = noisy_image_plus_masks_batch.to(device)
        batch_size = input_data.shape[0]
        optimizer.zero_grad()

        # Prediction denoising + mlp model (features and output)
        output_prediction = denoising_mlp_model(input_data)
        output_latent_features = output_prediction[0]  # Latent Features
        output_img_kernel = output_prediction[1]  # RGB image plus kernel
        output_variable = output_prediction[2]  # Label

        # Denoising Loss for RGB and Kernel
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

        # Stack Loss -- #  HERE
        denoising_loss_rgbk = denoising_loss_red + denoising_loss_green + denoising_loss_blue + denoising_loss_kernel

        denoising_loss_train = torch.stack([
            denoising_loss_red * weights[0],
            denoising_loss_green * weights[1],
            denoising_loss_blue * weights[2],
            denoising_loss_kernel * weights[3],
            loss_variable_mlp * weights[4],
        ], dim=0)

        total_loss = denoising_loss_train.sum()

        # Optimize Loss
        total_loss.backward()

        # Optimizer Steps
        optimizer.step()

        print('epoch', epoch)

        if batch_idx % config['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'.format(
                epoch, batch_idx * original_image_batch.shape[0], len(train_loader.dataset),
                       100. * batch_idx / len(train_loader)))

            print('Total Loss:{:.6f}'.format(total_loss.item()))
            print('Denoising Loss:{:.6f}'.format(denoising_loss_rgbk.item()))
            print('MLP Loss:{:.6f}'.format(loss_variable_mlp.item()))

    return total_loss, denoising_loss_rgbk, loss_variable_mlp


# Test Function
def test(config, denoising_mlp_model, device, val_loader,
         denoising_loss_dict, mlp_loss, epoch, weights=None,
         ):

    if weights is None:
        weights = [0.07, 0.07, 0.07, 0.07, 2.6]

    denoising_mlp_model.eval()
    total_loss_weighted, denoising_loss_test = None, None

    with (torch.no_grad()):
        # Test per batch
        for batch_idx, sample_batch in enumerate(val_loader):
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

            total_loss = denoising_loss_test.sum()

            print('epoch', epoch)

            if batch_idx % config['log_interval'] == 0:
                print('Val. Epoch: {} [{}/{} ({:.0f}%)]\t'.format(
                    epoch, batch_idx * original_image_batch.shape[0], len(val_loader.dataset),
                           100. * batch_idx / len(val_loader)))

                print('Val. Total Loss:{:.6f}'.format(total_loss.item()))
                print('Val. Denoising Loss:{:.6f}'.format(denoising_loss_rgbk.item()))
                print('Val. MLP Loss:{:.6f}'.format(loss_variable_mlp.item()))

    return total_loss, denoising_loss_rgbk, loss_variable_mlp
