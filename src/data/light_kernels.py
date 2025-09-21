import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def polynomial_kernel(dim_x, dim_y, n_power_x=1, n_power_y=1, scale_x=1, scale_y=1, offset_x=0, offset_y=0):
    """ Returns an image with a polynomial kernel noise distribution.

        Parameters
        ----------
        dim_x : int
            image height in pixels
        dim_y : int
            image width in pixels
        n_power_x, n_power_y : int, int
            power for the polynomial functions in x and y
        scale_x, scale_y : int, int
        offset_x, offset_y: offset of the polynomial function
    """
    # polynomial for x dimension
    kernel_x = np.zeros((dim_y, dim_x))
    x = np.arange(dim_x) / dim_x
    x_polynomial = ((scale_x * x) ** n_power_x) + offset_x
    for i in range(dim_y):
        kernel_x[i:i + 1, :] = x_polynomial

    # polynomial for y dimension
    kernel_y = np.zeros((dim_x, dim_y))
    y = np.arange(dim_y) / dim_y
    y_polynomial = ((scale_y * y) ** n_power_y) + offset_y
    for i in range(dim_x):
        kernel_y[i:i + 1, :] = y_polynomial
    kernel_y = np.transpose(kernel_y)

    # Average x & y kernels
    kernel_xy = np.mean(np.array([kernel_x, kernel_y]), axis=0)

    # Min-Max Normalization [0-1]
    if (np.nanmax(kernel_xy) - np.nanmin(kernel_xy)) > 0:
        kernel_xy = (kernel_xy - np.nanmin(kernel_xy)) / (np.nanmax(kernel_xy) - np.nanmin(kernel_xy))
    else:
        kernel_xy = (kernel_xy - np.nanmin(kernel_xy))

    # Remove nans in case
    kernel_xy = np.nan_to_num(kernel_xy)

    return kernel_xy


def radial_kernel(dim_x, dim_y, center_x, center_y, scale_radius_x, scale_radius_y):
    """ Returns an image with an elliptical kernel noise distribution.

        Parameters
        ----------
        dim_x : int
            image height in pixels
        dim_y : int
            image width in pixels
        center_x, center_y : int, int
            offset of the center of the elliptical distribution
        scale_radius_x, scale_radius_y : int, int
    """
    dist_to_center_x_array = [(((x - center_x) / dim_x) ** 2 * scale_radius_x) for x in np.arange(dim_x)]
    dist_to_center_y_array = [(((y - center_y) / dim_y) ** 2 * scale_radius_y) for y in np.arange(dim_y)]

    kernel_x = np.zeros((dim_y, dim_x))
    for i in range(dim_y):
        kernel_x[i:i + 1, :] = dist_to_center_x_array

    kernel_y = np.zeros((dim_x, dim_y))
    for i in range(dim_x):
        kernel_y[i:i + 1, :] = dist_to_center_y_array
    kernel_y = np.transpose(kernel_y)

    distance_kernel = kernel_x + kernel_y + np.ones(kernel_x.shape)
    light_kernel = np.divide(np.ones(distance_kernel.shape), distance_kernel)

    # Min-Max Normalization [0-1]
    if (np.nanmax(light_kernel) - np.nanmin(light_kernel)) > 0:
        kernel_xy = (light_kernel - np.nanmin(light_kernel)) / (np.nanmax(light_kernel) - np.nanmin(light_kernel))
    else:
        kernel_xy = (light_kernel - np.nanmin(light_kernel))

    # Remove nans in case
    light_kernel = np.nan_to_num(light_kernel)

    return light_kernel


def add_kernel_to_image(image_in, alpha, beta, gamma, kernel, delta=5):
    """ Returns the input image merged with a noise kernel.

        Parameters
        ----------
        image_in : image
            input image to be modified
        alpha, beta, gamma : int, int, int
            intensity factors to merge images
        kernel : image
            noise distribution
    """
    test_image_in = image_in.copy()
    img_out = np.zeros(image_in.shape)

    alpha_m = kernel * alpha
    beta_m = kernel * beta
    gamma_m = kernel * gamma

    array_kernels = [alpha_m, beta_m, gamma_m]

    for i in range(3):
        image_plus_kernel = test_image_in[:, :, i] + array_kernels[i]
        max_val_img_kernel = np.max(image_plus_kernel)
        min_val_img_kernel = np.min(image_plus_kernel)

        image_plus_kernel_final = image_plus_kernel

        if max_val_img_kernel > (255 - delta):
            mask_vals_above = image_plus_kernel > (255-delta)
            array_vals_above = mask_vals_above * (255-delta)

            mask_vals_below = image_plus_kernel < 255-delta
            image_plus_kernel_final = (image_plus_kernel * mask_vals_below) + array_vals_above

        elif min_val_img_kernel < delta:
            mask_vals_below = image_plus_kernel < delta
            array_vals_below = mask_vals_below * delta

            mask_vals_above = image_plus_kernel > delta
            image_plus_kernel_final = (image_plus_kernel * mask_vals_above) + array_vals_below

        img_out[:, :, i] = image_plus_kernel_final

    img_out = np.clip(img_out, 0, 255)
    img_out_u8 = img_out.astype(np.uint8)
    return img_out_u8


def rotate_flip(input_img):
    """ Returns the input image randomly rotated or flipped.
    """
    rotate_or_flip = random.randint(0, 2)
    if rotate_or_flip == 0:
        flip_ud_lr = random.randint(0, 1)
        if flip_ud_lr == 0:
            input_img = np.flipud(input_img)
        else:
            input_img = np.fliplr(input_img)
    elif rotate_or_flip == 1:
        rot_c = random.randint(0, 2)
        if rot_c == 0:
            input_img = np.rot90(input_img)
        if rot_c == 1:
            input_img = np.rot90(input_img)
            input_img = np.rot90(input_img)
        else:
            input_img = np.rot90(input_img)
            input_img = np.rot90(input_img)
            input_img = np.rot90(input_img)
    return input_img


def random_sample_kernel(kernel_type, img_shape):
    """ Returns a kernel with randomly sampled parameters.
    """
    sampled_kernel = None
    if kernel_type == 'polynomial':
        polynom_array = np.array([1, 2, 3, 4])
        linear_scale = np.arange(-10, 10, 1)
        linear_offset = np.arange(-20, 20, 1)

        polynomial_light_kernel = polynomial_kernel(
            dim_x=img_shape[0],
            dim_y=img_shape[1],
            n_power_x=np.random.choice(polynom_array, 1)[0],
            n_power_y=np.random.choice(polynom_array, 1)[0],
            scale_x=np.random.choice(linear_scale, 1)[0],
            scale_y=np.random.choice(linear_scale, 1)[0],
            offset_x=np.random.choice(linear_offset, 1)[0],
            offset_y=np.random.choice(linear_offset, 1)[0],
        )
        polynomial_light_kernel = rotate_flip(polynomial_light_kernel)
        sampled_kernel = polynomial_light_kernel

    elif kernel_type == 'radial':
        center_xy = np.arange(-int(img_shape[0] / 2), img_shape[0], 100)
        scale_radius = np.arange(1, 40, 0.5)

        radial_light_kernel = radial_kernel(
            dim_x=img_shape[0],
            dim_y=img_shape[1],
            center_x=np.random.choice(center_xy, 1)[0],
            center_y=np.random.choice(center_xy, 1)[0],
            scale_radius_x=np.random.choice(scale_radius, 1)[0],
            scale_radius_y=np.random.choice(scale_radius, 1)[0],
        )
        radial_light_kernel = rotate_flip(radial_light_kernel)
        sampled_kernel = radial_light_kernel

    return sampled_kernel


def add_noise_to_image(image_in, outer_mask_ref, highest_noise_level=0.5):
    # noise levels and sign
    alpha_beta_gamma = np.arange(0, highest_noise_level, 0.01) * 255
    temp_sign = np.random.choice([1, -1], 1)[0]

    # sample a kernel
    kernel_type_sample = random.sample(['polynomial', 'radial'], 1)[0]
    sampled_kernel = random_sample_kernel(
        kernel_type=kernel_type_sample,
        img_shape=image_in.shape
    )

    # Sample Noise
    sampled_alpha = int(np.random.choice(alpha_beta_gamma, 1)[0]) * temp_sign
    sampled_beta = int(np.random.choice(alpha_beta_gamma, 1)[0]) * temp_sign
    sampled_gamma = int(np.random.choice(alpha_beta_gamma, 1)[0]) * temp_sign

    temp_img_in = image_in.copy()

    noise_image = add_kernel_to_image(
        image_in=temp_img_in,
        alpha=sampled_alpha,
        beta=sampled_beta,
        gamma=sampled_gamma,
        kernel=sampled_kernel,
    )
    noise_image[outer_mask_ref] = 0
    sampled_kernel[outer_mask_ref] = 0

    noise_params = {
        'sampled_kernel': kernel_type_sample,
        'sampled_alpha': sampled_alpha,
        'sampled_beta': sampled_beta,
        'sampled_gamma': sampled_gamma,
    }

    return noise_image, sampled_kernel, noise_params
