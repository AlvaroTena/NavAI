import os

import neptune
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tf_agents.specs import array_spec

import pewrapper.types.constants as pe_const
import rlnav.types.constants as const
from navutils.logger import Logger
from pewrapper.managers import ConfigurationManager
from rlnav.data.dataset import RLDataset
from rlnav.models import vivit
from rlnav.types import vivit_utils as vutils

run = neptune.init_run(
    project="AI-PE/RL-GSharp",
    monitoring_namespace="monitoring/04_obs_encoding",
    source_files=["config/RLNav/params.yaml", "src/rlnav/**/*.py"],
)
if run.exists("sys/failed"):
    del run["sys/failed"]

Logger("./").set_category("DEBUG")

config = ConfigurationManager("./", "")
config.parse_config_file(
    "../subprojects/common_lib/config_PE_jenkins/configs_ai_qm/config_F9P_L1L2_BDS.txt"
)

time_steps = 100
rldataset = RLDataset(
    config_signals=[
        [
            config.config_info_.Signal_1_GPS,
            config.config_info_.Signal_2_GPS,
        ],
        [
            config.config_info_.Signal_1_GAL,
            config.config_info_.Signal_2_GAL,
        ],
        [
            config.config_info_.Signal_1_BDS,
            config.config_info_.Signal_2_BDS,
        ],
    ],
    transformer_path="data/RLNav/transformed/transform_fn.pkl",
    window_size=time_steps,
)
rldataset.read_data(
    os.path.join(
        "data/scenarios/",
        "scenario_931_000_GPSGALBDS_CSIRELAND_L1L2_KINEMATIC_20231003/INPUTS/AI/AI_Multipath_20231003_092730.txt",
    )
)

transformed_dataset = rldataset.transform_features()
obs_min = transformed_dataset.min().values
obs_max = transformed_dataset.max().values


scaler = StandardScaler()
standard_data = scaler.fit_transform(
    transformed_dataset[
        [
            "delta_time",
            "code",
            "phase",
            "doppler",
            "snr",
            "elevation",
            "residual",
            "iono",
            "delta_cmc",
            "crc",
        ]
    ]
)
transformed_dataset.loc[
    :,
    [
        "delta_time",
        "code",
        "phase",
        "doppler",
        "snr",
        "elevation",
        "residual",
        "iono",
        "delta_cmc",
        "crc",
    ],
] = standard_data


min_values, max_values = (
    transformed_dataset.min().to_list(),
    transformed_dataset.max().to_list(),
)
obs_spec = array_spec.BoundedArraySpec(
    shape=(
        pe_const.MAX_SATS * pe_const.NUM_CHANNELS,
        1 + len(const.PROCESSED_FEATURE_LIST),
    ),
    dtype=np.float32,
    minimum=np.array([0.0] + min_values),
    maximum=np.array([1.0] + max_values),
    name="observation",
)

window_size = rldataset.window_size
height = pe_const.MAX_SATS * pe_const.NUM_CHANNELS
width = len(const.PROCESSED_FEATURE_LIST)
channels = 1
factorised_model = vivit.ViViT_Factorised(
    spatial_patch_size=(height // height, width // 1),
    embedding_dim=1024,
    window_size=window_size,
    height=height,
    width=width,
    channels=channels,
    num_layers_spatial=1,
    num_layers_temporal=1,
)
factorised_model.build_model()

# Definir el modelo
autoencoder = factorised_model.get_autoencoder()
encoder = factorised_model.get_encoder()


def masked_mse_loss(y_true, y_pred):
    """
    Calcula la pérdida MSE enmascarada.

    Args:
        y_true: Tensor con forma (batch_size, time_steps, H, W, C), con la máscara.
        y_pred: Tensor con forma (batch_size, time_steps, H, W, C).

    Returns:
        Pérdida MSE promedio considerando solo las posiciones válidas.
    """
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.squeeze(y_pred, axis=-1)

    # Extraer la máscara y las características reales
    mask = y_true[..., 0:1]  # Máscara de atención (Shape: batch_size, T, H, W)
    y_true = y_true[..., 1:]  # Características reales (Shape: batch_size, T, H, W)

    # Calcular el error cuadrático medio (MSE)
    mse = tf.square(y_true - y_pred)

    # Aplicar la máscara de validez
    masked_mse = mse * mask

    # Calcular el promedio solo sobre las posiciones válidas
    masked_mse_sum = tf.reduce_sum(masked_mse)
    validity_mask_sum = tf.reduce_sum(mask)

    # Evitar la división por cero
    epsilon = 1e-9
    masked_mse_mean = masked_mse_sum / (validity_mask_sum + epsilon)

    return masked_mse_mean


def generator():
    def wrapper_transform(block_data):
        return rldataset.transform_to_observation_spec(block_data, obs_spec)

    return vutils.data_generator(
        transformed_dataset, (window_size, height, width, channels), wrapper_transform
    )


dataset = tf.data.Dataset.from_generator(
    generator,
    output_types=(
        {"input_observations": tf.float32, "attention_mask": tf.int32},
        tf.float32,
    ),
    output_shapes=(
        {
            "input_observations": (window_size, height, width, channels),
            "attention_mask": (window_size, height, width, channels),
        },
        (window_size, height, width + 1, channels),
    ),
)

sample = dataset.take(1)
for inputs, output in sample:
    sample = (inputs, output)


total_epochs = len(transformed_dataset.index.get_level_values("epoch").unique())
steps_per_epoch = total_epochs // time_steps
batch_size = 1
buffer_size = steps_per_epoch

dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)


# Compilar el modelo
autoencoder.compile(
    optimizer=tf.keras.optimizers.AdamW(),
    loss=masked_mse_loss,
)

# Expandir obs_min y obs_max para que tengan la forma adecuada para la operación de broadcasting
obs_min_exp = np.expand_dims(obs_min, axis=(0, 2))  # De (14,) a (1, 14, 1)
obs_max_exp = np.expand_dims(obs_max, axis=(0, 2))  # De (14,) a (1, 14, 1)


ground_truth = sample[0]["input_observations"]
mask = sample[0]["attention_mask"]

# Expandir dimensiones para cumplir con el input requerido por el modelo
input_observations = tf.expand_dims(ground_truth, axis=0)  # Expande el batch dimension
attention_mask = tf.expand_dims(mask, axis=0)  # Expande el batch dimension

# Predicción del modelo
predicted_obs = autoencoder.predict(
    {
        "input_observations": input_observations,
        "attention_mask": attention_mask,
    }
)

# Visualizar la secuencia
gif_path = vutils.generate_sequence_comparison_gif(
    ground_truth.numpy(),
    predicted_obs,
    mask.numpy(),
    reverse_scaling=scaler.inverse_transform,
    obs_min=obs_min_exp,
    obs_max=obs_max_exp,
    row_labels=transformed_dataset.columns.to_list(),
    save_path="pretrain.gif",
)
run[f"visualizations/pretrain"].upload(gif_path)


# Definir el número de épocas y pasos por época
num_epochs = 5

# Entrenar el modelo
autoencoder.fit(
    dataset.repeat(steps_per_epoch * num_epochs),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
)

autoencoder.save("models/autoencoder.keras")
run["autoencoder"].upload("models/autoencoder.keras")
gif_path = vutils.generate_sequence_comparison_gif(
    ground_truth.numpy(),
    predicted_obs,
    mask.numpy(),
    reverse_scaling=scaler.inverse_transform,
    obs_min=obs_min_exp,
    obs_max=obs_max_exp,
    row_labels=transformed_dataset.columns.to_list(),
    save_path="trained.gif",
)
run[f"visualizations/trained"].upload(gif_path)

Logger.reset()
run.stop()
