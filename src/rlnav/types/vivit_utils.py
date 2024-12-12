import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation

import rlnav.types.constants as const


def data_generator(dataset, data_shape, transform_fn):
    T, H, W, C = data_shape
    unique_epochs = dataset.index.get_level_values("epoch").unique()
    total_epochs = len(unique_epochs)

    for start_idx in range(0, total_epochs, T):
        end_idx = min(start_idx + T, total_epochs)
        selected_epochs = unique_epochs[start_idx:end_idx]
        block_data = dataset[
            dataset.index.get_level_values("epoch").isin(selected_epochs)
        ]

        obs = transform_fn(block_data)
        attention_mask = (obs[..., 0]).astype(np.int32)  # Extract attention mask
        obs = obs[..., 1:]  # Extract features

        y_true = np.concatenate([np.expand_dims(attention_mask, axis=-1), obs], axis=-1)

        # Convert to tensors
        obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
        attention_mask_tensor = tf.convert_to_tensor(attention_mask, dtype=tf.int32)
        y_true_tensor = tf.convert_to_tensor(y_true, dtype=tf.float32)

        # Adjust shapes
        obs_tensor = tf.expand_dims(obs_tensor, axis=-1)  # Add channel dimension
        y_true_tensor = tf.expand_dims(y_true_tensor, axis=-1)
        attention_mask_expanded = tf.expand_dims(
            attention_mask_tensor, axis=-1
        )  # Add channel dimension
        attention_mask_expanded = tf.tile(
            attention_mask_expanded, [1, 1, W]
        )  # Repeat along width
        attention_mask_expanded = tf.expand_dims(
            attention_mask_expanded, axis=-1
        )  # Add channel dimension

        # Prepare inputs and outputs for the model
        inputs = {
            "input_observations": obs_tensor,
            "attention_mask": attention_mask_expanded,
        }
        outputs = y_true_tensor

        yield inputs, outputs


def generate_sequence_comparison_gif(
    ground_truth,
    predictions,
    attention_mask,
    reverse_scaling,
    obs_min,
    obs_max,
    row_labels,
    num_frames=100,
    save_path="sequence.gif",
):
    """
    Generates a GIF comparing ground truth observations, predictions, and their differences.

    Args:
        ground_truth (np.ndarray): Array of ground truth observations. Shape: (batch_size, time_steps, width, height, channels).
        predictions (np.ndarray): Array of predicted observations. Shape: (batch_size, time_steps, width, height, channels).
        attention_mask (np.ndarray): Attention mask to apply to the observations. Shape: (batch_size, time_steps, width, height, channels).
        reverse_scaling (callable): Function to reverse the scaling transformation applied to the observations.
        obs_min (float): Minimum value for observation normalization.
        obs_max (float): Maximum value for observation normalization.
        row_labels (list of str): Labels for each row in the visualized sequence.
        num_frames (int): Number of frames in the animation. Default is 100.
        save_path (str): Path to save the resulting GIF. Default is "sequence.gif".

    Returns:
        str: Path to the saved GIF file.
    """

    # Remove batch and channel dimensions
    predictions = np.squeeze(
        predictions, axis=(0, -1)
    )  # Shape: (time_steps, width, height)
    ground_truth = np.squeeze(
        ground_truth, axis=-1
    )  # Shape: (time_steps, width, height)
    attention_mask = np.squeeze(
        attention_mask, axis=-1
    )  # Shape: (time_steps, width, height)

    # Reshape and reverse scaling for specific channels
    data_shape = ground_truth.shape
    ground_truth = ground_truth.reshape(-1, data_shape[-1])
    predictions = predictions.reshape(-1, data_shape[-1])

    n_timestamps_features = len(const.PROCESSED_TIMESERIES_LIST)
    ground_truth[..., n_timestamps_features:] = reverse_scaling(
        ground_truth[..., n_timestamps_features:]
    )
    predictions[..., n_timestamps_features:] = reverse_scaling(
        predictions[..., n_timestamps_features:]
    )

    ground_truth = ground_truth.reshape(data_shape)
    predictions = predictions.reshape(data_shape)

    # Transpose arrays to align with visualization (time_steps, height, width)
    ground_truth = np.transpose(ground_truth, (0, 2, 1))
    predictions = np.transpose(predictions, (0, 2, 1))
    attention_mask = np.transpose(attention_mask, (0, 2, 1))

    # Calculate difference
    difference = np.abs(ground_truth - predictions)

    # Normalize data to [0, 1] range
    ground_truth = np.clip((ground_truth - obs_min) / (obs_max - obs_min + 1e-9), 0, 1)
    predictions = np.clip((predictions - obs_min) / (obs_max - obs_min + 1e-9), 0, 1)
    diff_min = np.min(difference, axis=(0, 2), keepdims=True)
    diff_max = np.max(difference, axis=(0, 2), keepdims=True)
    difference = np.clip((difference - diff_min) / (diff_max - diff_min + 1e-9), 0, 1)

    # Scale to 255 for visualization
    ground_truth *= 255
    predictions *= 255
    difference *= 255

    # Apply attention mask
    ground_truth *= attention_mask
    predictions *= attention_mask
    difference *= attention_mask

    # Initialize figure
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle("Ground Truth vs Prediction vs Difference")

    ax_gt, ax_pred, ax_diff = axes
    im_gt = ax_gt.imshow(
        ground_truth[0],
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        aspect="auto",
    )
    im_pred = ax_pred.imshow(
        predictions[0],
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        aspect="auto",
    )
    im_diff = ax_diff.imshow(
        difference[0],
        cmap="hot",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        aspect="auto",
    )

    # Add labels and grid
    for ax, title in zip(axes, ["Ground Truth", "Prediction", "Difference"]):
        ax.set_title(title)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=12)
        ax.grid(True, which="both", color="black", linestyle="-", linewidth=0.5)

    # Update function for animation
    def update_frame(frame):
        im_gt.set_array(ground_truth[frame])
        im_pred.set_array(predictions[frame])
        im_diff.set_array(difference[frame])
        return im_gt, im_pred, im_diff

    # Create animation
    animation = FuncAnimation(
        fig, update_frame, frames=num_frames, interval=200, blit=False
    )

    # Save animation as GIF
    gif_full_path = os.path.join("reports/", save_path)
    animation.save(gif_full_path, writer="pillow", fps=5)

    plt.close(fig)
    return gif_full_path


def create_tubelet_embedding(input_data, attention_mask, tubelet_size, hidden_size):
    tubelet_size_t, tubelet_size_h, tubelet_size_w = tubelet_size

    # Get static and dynamic input shapes
    input_shape_static = input_data.shape
    batch_size, T, H, W, C = input_shape_static
    input_shape = tf.shape(input_data)
    if batch_size is None:
        batch_size = input_shape[0]  # Use dynamic batch size if static is None

    # Apply attention mask to remove padding influence
    input_data *= tf.cast(attention_mask, tf.float32)

    # Extract spatiotemporal features using a 3D convolution
    conv_layer = tf.keras.layers.Conv3D(
        filters=hidden_size,
        kernel_size=(tubelet_size_t, tubelet_size_h, tubelet_size_w),
        strides=(tubelet_size_t, tubelet_size_h, tubelet_size_w),
        padding="valid",
    )
    tubelets = conv_layer(
        input_data
    )  # Shape: (batch_size, num_tubelets_t, num_tubelets_h, num_tubelets_w, hidden_size)

    # Reduce attention mask to tubelet level
    mask_tubelets = tf.reshape(
        attention_mask,
        (
            batch_size,
            T // tubelet_size_t,
            tubelet_size_t,
            H // tubelet_size_h,
            tubelet_size_h,
            W // tubelet_size_w,
            tubelet_size_w,
            C,
        ),
    )
    mask_tubelets = tf.reduce_mean(
        mask_tubelets, axis=[2, 4, 6, 7]
    )  # Shape: (batch_size, num_tubelets_t, num_tubelets_h, num_tubelets_w)
    mask_tubelets = tf.cast(mask_tubelets > 0, tf.float32)

    # Apply mask to convolution output
    tubelets *= tf.expand_dims(mask_tubelets, axis=-1)

    # Flatten tubelets into a sequence
    num_tubelets = tubelets.shape[1] * tubelets.shape[2] * tubelets.shape[3]
    tubelets = tf.reshape(tubelets, [batch_size, num_tubelets, hidden_size])

    # Flatten mask for tubelet sequence
    mask_tubelets = tf.reshape(mask_tubelets, [batch_size, num_tubelets])

    return num_tubelets, tubelets, mask_tubelets


def create_spatial_patches(input_data, attention_mask, spatial_patch_size, hidden_size):
    spatial_patch_size_h, spatial_patch_size_w = spatial_patch_size

    # Get static and dynamic input shapes
    input_shape_static = input_data.shape
    batch_size, T, H, W, C = input_shape_static
    input_shape = tf.shape(input_data)
    if batch_size is None:
        batch_size = input_shape[0]  # Use dynamic batch size if static is None

    # Reshape per frames. Shape: (batch_size * T, H, W, C)
    input_frames = tf.reshape(input_data, (-1, H, W, C))
    attention_frames = tf.reshape(attention_mask, (-1, H, W, C))

    # Apply attention mask to remove padding influence
    input_frames *= tf.cast(attention_frames, tf.float32)

    # Extract spatial features using a 2D convolution
    conv_layer = tf.keras.layers.Conv2D(
        filters=hidden_size,
        kernel_size=(spatial_patch_size_h, spatial_patch_size_w),
        strides=(spatial_patch_size_h, spatial_patch_size_w),
        padding="valid",
    )
    spatial_patches = conv_layer(
        input_frames
    )  # Shape: (batch_size * T, num_patches_h, num_patches_w, hidden_size)

    # Reduce attention mask to spatial_patches level
    mask_patches = tf.reshape(
        attention_frames,
        (
            batch_size * T,
            H // spatial_patch_size_h,
            spatial_patch_size_h,
            W // spatial_patch_size_w,
            spatial_patch_size_w,
            C,
        ),
    )
    mask_patches = tf.reduce_mean(
        mask_patches, axis=[2, 4, 5]
    )  # Shape: (batch_size, num_patches_h, num_patches_w)
    mask_patches = tf.cast(mask_patches > 0, tf.float32)

    # Apply mask to convolution output
    spatial_patches *= tf.expand_dims(mask_patches, axis=-1)

    # Flatten spatial_patches into a sequence of tokens per frame
    num_patches = spatial_patches.shape[1] * spatial_patches.shape[2]
    spatial_patches = tf.reshape(
        spatial_patches, [batch_size, T, num_patches, hidden_size]
    )

    # Flatten mask into a sequence of tokens per frame
    mask_patches = tf.reshape(mask_patches, [batch_size, T, num_patches])

    return num_patches, spatial_patches, mask_patches
