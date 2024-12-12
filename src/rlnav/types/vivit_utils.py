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

        obs_channel_1, obs_channel_2 = np.split(obs, C, axis=1)
        obs = np.concatenate(
            [
                np.expand_dims(obs_channel_1, axis=-1),
                np.expand_dims(obs_channel_2, axis=-1),
            ],
            axis=-1,
        )
        assert obs.shape == data_shape

        attention_mask = np.expand_dims(attention_mask, axis=-1)
        attention_mask_channel_1, attention_mask_channel_2 = np.split(
            attention_mask, C, axis=1
        )
        attention_mask = np.concatenate(
            [
                np.expand_dims(attention_mask_channel_1, axis=-1),
                np.expand_dims(attention_mask_channel_2, axis=-1),
            ],
            axis=-1,
        )
        attention_mask = np.tile(attention_mask, (1, 1, W, 1))
        assert attention_mask.shape == (T, H, W, C)

        y_true = np.concatenate(
            [np.expand_dims(attention_mask[..., 0, :], axis=-2), obs], axis=-2
        )
        assert y_true.shape == (T, H, W + 1, C)

        # Convert to tensors
        obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
        attention_mask_tensor = tf.convert_to_tensor(attention_mask, dtype=tf.int32)
        y_true_tensor = tf.convert_to_tensor(y_true, dtype=tf.float32)

        # Prepare inputs and outputs for the model
        inputs = {
            "input_observations": obs_tensor,
            "attention_mask": attention_mask_tensor,
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
    combine_channels=False,
):
    """
    Generates a GIF comparing ground truth observations, predictions, and their differences for 2 channels.

    Args:
        ground_truth (np.ndarray): Ground truth data. Shape: (time_steps, width, height, channels).
        predictions (np.ndarray): Predicted data. Shape: (batch_size, time_steps, width, height, channels).
        attention_mask (np.ndarray): Attention mask. Shape: (time_steps, width, height, channels).
        reverse_scaling (callable): Function to reverse scaling transformations.
        obs_min (float): Minimum observation value for normalization.
        obs_max (float): Maximum observation value for normalization.
        row_labels (list of str): Labels for rows in visualized sequence.
        num_frames (int): Number of frames in the animation.
        save_path (str): Path to save the resulting GIF.
        combine_channels (bool): If True, combines the two channels into one visualization.

    Returns:
        str: Path to the saved GIF file.
    """

    T, H, W, C = ground_truth.shape
    # Remove batch dimension
    predictions = np.squeeze(predictions, axis=0)

    # Reverse scaling
    ground_truth = ground_truth.transpose(0, 1, 3, 2)
    ground_truth = ground_truth.reshape(-1, W)
    predictions = predictions.transpose(0, 1, 3, 2)
    predictions = predictions.reshape(-1, W)

    ground_truth = reverse_scaling(ground_truth)
    predictions = reverse_scaling(predictions)

    ground_truth = ground_truth.reshape(T, H, C, W)
    ground_truth = ground_truth.transpose(0, 1, 3, 2)
    predictions = predictions.reshape(T, H, C, W)
    predictions = predictions.transpose(0, 1, 3, 2)

    # Normalize data
    ground_truth = np.clip((ground_truth - obs_min) / (obs_max - obs_min + 1e-9), 0, 1)
    predictions = np.clip((predictions - obs_min) / (obs_max - obs_min + 1e-9), 0, 1)
    difference = np.abs(ground_truth - predictions)
    difference = np.clip(
        (difference - np.min(difference)) / (np.max(difference) + 1e-9), 0, 1
    )
    ground_truth *= 255
    predictions *= 255
    difference *= 255

    # Apply attention mask
    ground_truth *= attention_mask
    predictions *= attention_mask
    difference *= attention_mask

    ground_truth = ground_truth.transpose(0, 2, 1, 3)
    predictions = predictions.transpose(0, 2, 1, 3)
    difference = difference.transpose(0, 2, 1, 3)

    # Initialize figure
    fig, axes = plt.subplots(2 if not combine_channels else 1, 3, figsize=(30, 20))
    fig.suptitle("Ground Truth vs Prediction vs Difference")

    # Split channels for visualization
    channel_1_gt, channel_2_gt = ground_truth[..., 0], ground_truth[..., 1]
    channel_1_pred, channel_2_pred = predictions[..., 0], predictions[..., 1]
    channel_1_diff, channel_2_diff = difference[..., 0], difference[..., 1]

    # Combine channels with a 2-channel color scheme (R=channel_1, G=channel_2)
    if combine_channels:
        ground_truth_combined = np.stack(
            [channel_1_gt, channel_2_gt, np.zeros_like(channel_1_gt)], axis=-1
        )  # R=channel_1, G=channel_2, B=0
        predictions_combined = np.stack(
            [channel_1_pred, channel_2_pred, np.zeros_like(channel_1_pred)], axis=-1
        )  # R=channel_1, G=channel_2, B=0
        difference_combined = np.stack(
            [channel_1_diff, channel_2_diff, np.zeros_like(channel_1_diff)], axis=-1
        )  # R=channel_1, G=channel_2, B=0

    # Initialize plots
    if combine_channels:
        for ax, title in zip(axes, ["Ground Truth", "Prediction", "Difference"]):
            ax.set_title(title)
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=12)
            ax.grid(True, which="both", color="black", linestyle="-", linewidth=0.5)

        ax_gt, ax_pred, ax_diff = axes
        im_gt = ax_gt.imshow(
            ground_truth_combined[0, :, :, :],
            vmin=0,
            vmax=255,
            interpolation="nearest",
            aspect="auto",
        )
        im_pred = ax_pred.imshow(
            predictions_combined[0, :, :, :],
            vmin=0,
            vmax=255,
            interpolation="nearest",
            aspect="auto",
        )
        im_diff = ax_diff.imshow(
            difference_combined[0, :, :, :],
            vmin=0,
            vmax=255,
            interpolation="nearest",
            aspect="auto",
        )
    else:
        for i, (data_gt, data_pred, data_diff, ax_row) in enumerate(
            [
                (channel_1_gt, channel_1_pred, channel_1_diff, axes[0]),
                (channel_2_gt, channel_2_pred, channel_2_diff, axes[1]),
            ]
        ):
            for ax, title in zip(
                ax_row,
                [
                    f"Ground Truth - Channel {i+1}",
                    f"Prediction - Channel {i+1}",
                    f"Difference - Channel {i+1}",
                ],
            ):
                ax.set_title(title)
                ax.set_yticks(np.arange(len(row_labels)))
                ax.set_yticklabels(row_labels, fontsize=12)
                ax.grid(True, which="both", color="black", linestyle="-", linewidth=0.5)

            ax_gt, ax_pred, ax_diff = ax_row
            im_gt = ax_gt.imshow(
                data_gt[0, :, :],
                cmap="gray",
                vmin=0,
                vmax=255,
                interpolation="nearest",
                aspect="auto",
            )
            im_pred = ax_pred.imshow(
                data_pred[0, :, :],
                cmap="gray",
                vmin=0,
                vmax=255,
                interpolation="nearest",
                aspect="auto",
            )
            im_diff = ax_diff.imshow(
                data_diff[0, :, :],
                cmap="hot",
                vmin=0,
                vmax=255,
                interpolation="nearest",
                aspect="auto",
            )

    # Update function for animation
    def update_frame(frame):
        if combine_channels:
            im_gt.set_array(ground_truth_combined[frame, :, :, :])
            im_pred.set_array(predictions_combined[frame, :, :, :])
            im_diff.set_array(difference_combined[frame, :, :, :])
        else:
            im_gt.set_array(channel_1_gt[frame, :, :])
            im_pred.set_array(channel_1_pred[frame, :, :])
            im_diff.set_array(channel_1_diff[frame, :, :])
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
