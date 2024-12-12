import os

import matplotlib.pyplot as plt
import neptune
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import StandardScaler
from tf_agents.specs import array_spec

import pewrapper.types.constants as pe_const
import rlnav.types.constants as const
from navutils.logger import Logger
from pewrapper.managers import ConfigurationManager
from rlnav.data.dataset import RLDataset

run = neptune.init_run(
    project="AI-PE/RL-GSharp",
    monitoring_namespace="monitoring/04_obs_encoding",
    source_files=["config/RLNav/params.yaml", "src/rlnav/**/*.py"],
)


Logger("./").set_category("DEBUG")

config = ConfigurationManager("./", "")
config.parse_config_file(
    "../subprojects/common_lib/config_PE_jenkins/configs_ai_qm/config_F9P_L1L2_BDS.txt"
)

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

time_steps = 100
rldataset.window_size = time_steps

# Parámetros de los parches
patch_size_t = 10  # Tamaño de parche temporal
patch_size_h = 13  # Tamaño de parche en altura (286 / 13 = 22 parches)
patch_size_w = 14  # Tamaño de parche en anchura (14 / 14 = 1 parches)
T = time_steps
H = 286
W = 14
C = 1


def data_generator(window_size=time_steps):
    # Obtener la lista de valores únicos en el nivel "epoch"
    unique_epochs = transformed_dataset.index.get_level_values("epoch").unique()
    total_epochs = len(unique_epochs)

    # Iterar en bloques de tamaño window_size
    for start_idx in range(0, total_epochs, window_size):
        end_idx = min(start_idx + window_size, total_epochs)

        selected_epochs = unique_epochs[start_idx:end_idx]
        block_data = transformed_dataset[
            transformed_dataset.index.get_level_values("epoch").isin(selected_epochs)
        ]

        obs = rldataset.transform_to_observation_spec(block_data, obs_spec)
        attention_mask = (obs[..., 0]).astype(np.int32)

        obs = obs[..., 1:]

        # Concatenar la máscara con las características reales en el eje de características
        y_true = np.concatenate([np.expand_dims(attention_mask, axis=-1), obs], axis=-1)

        # Convertir obs, attention_mask e y_true a tensores
        obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
        attention_mask_tensor = tf.convert_to_tensor(attention_mask, dtype=tf.int32)
        y_true_tensor = tf.convert_to_tensor(y_true, dtype=tf.float32)

        # Reshape obs_tensor para añadir la dimensión de canales
        obs_tensor = tf.expand_dims(obs_tensor, axis=-1)  # Shape: (100, 286, 14, 1)
        y_true_tensor = tf.expand_dims(
            y_true_tensor, axis=-1
        )  # Shape: (100, 286, 15, 1)

        # Expandir attention_mask_tensor para que tenga las mismas dimensiones que obs_tensor
        attention_mask_expanded = tf.expand_dims(
            attention_mask_tensor, axis=-1
        )  # Shape: (batch_size, time_steps, 286, 1)

        # Repetir a lo largo de la dimensión de características (W)
        attention_mask_expanded = tf.tile(
            attention_mask_expanded, [1, 1, W]
        )  # Shape: (batch_size, time_steps, 286, 14)

        # Añadir la dimensión de canales (C)
        attention_mask_expanded = tf.expand_dims(
            attention_mask_expanded, axis=-1
        )  # Shape: (batch_size, time_steps, 286, 14, 1)

        # Genera la entrada y salida para el modelo
        inputs = {
            "input_observations": obs_tensor,
            "attention_mask": attention_mask_expanded,
        }
        outputs = y_true_tensor

        yield inputs, outputs


def create_patches(
    input_data, attention_mask, patch_size_t, patch_size_h, patch_size_w, hidden_size
):
    # input_data tiene forma (batch_size, T, H, W, C)

    # Obtener forma estática para manejar dimensiones desconocidas
    input_shape_static = input_data.shape
    batch_size = input_shape_static[0]  # Puede ser None

    # Si batch_size es None, usar tf.shape() para obtener tamaño dinámico
    input_shape = tf.shape(input_data)
    if batch_size is None:
        batch_size = input_shape[0]  # Obtener tamaño dinámico del batch

    # Ajustar la máscara de atención para que tenga las mismas dimensiones que input_data
    attention_mask = tf.broadcast_to(
        attention_mask, [batch_size, T, H, W, C]
    )  # Shape: (batch_size, T, H, W, C)

    # Aplicar la máscara de atención para evitar el padding antes de la convolución
    input_data *= tf.cast(attention_mask, tf.float32)

    # Aplicar una convolución 3D para extraer características espacio-temporales
    conv_layer = tf.keras.layers.Conv3D(
        filters=hidden_size,  # Número de filtros que queremos, corresponde al "embedding dimension"
        kernel_size=(
            patch_size_t,
            patch_size_h,
            patch_size_w,
        ),  # Tamaño de los tubelets
        strides=(
            patch_size_t,
            patch_size_h,
            patch_size_w,
        ),  # Desplazamiento igual al tamaño del parche
        padding="valid",  # No se aplica padding adicional
    )
    x = conv_layer(
        input_data
    )  # Salida tiene forma (batch_size, num_patches_t, num_patches_h, num_patches_w, hidden_size)

    # Ajustar la máscara de atención para aplicar después de la convolución
    # La máscara también debe ser reducida al tamaño de los parches
    mask_patches = tf.reshape(
        attention_mask,
        (
            batch_size,
            T // patch_size_t,
            patch_size_t,
            H // patch_size_h,
            patch_size_h,
            W // patch_size_w,
            patch_size_w,
            C,
        ),
    )
    mask_patches = tf.reduce_mean(
        mask_patches, axis=[2, 4, 6, 7]
    )  # Reduce para obtener una máscara a nivel de parche
    mask_patches = tf.cast(
        mask_patches > 0, tf.float32
    )  # Convertir a 1 si alguna parte del parche es válida

    # Aplicar la máscara de atención después de la convolución para eliminar el efecto del padding
    x *= tf.expand_dims(mask_patches, axis=-1)

    # Reorganizar para obtener una secuencia de parches
    num_patches = x.shape[1] * x.shape[2] * x.shape[3]
    patches = tf.reshape(
        x, [batch_size, num_patches, hidden_size]
    )  # Salida tiene forma (batch_size, num_patches, hidden_size)

    # Convertir mask_patches en una máscara con forma (batch_size, num_patches, num_patches)
    mask_patches = tf.reshape(
        mask_patches, [batch_size, num_patches]
    )  # Shape: (batch_size, num_patches)

    return num_patches, patches, mask_patches


embedding_dim = 128

# Entrada del modelo
input_data = tf.keras.Input(shape=(time_steps, H, W, C), name="input_observations")
attention_mask_input = tf.keras.Input(
    shape=(time_steps, H, W, C), name="attention_mask"
)

input_shape = tf.shape(input_data)
batch_size = input_shape[0]

# Crear los parches
num_patches, patches, mask_patches = create_patches(
    input_data,
    attention_mask_input,
    patch_size_t,
    patch_size_h,
    patch_size_w,
    embedding_dim,
)  # Shape: (batch_size, num_patches, embedding_dim)


# Proyectar los parches a embeddings
patch_embeddings = tf.keras.layers.Dense(embedding_dim)(
    patches
)  # Shape: (batch_size, num_patches, embedding_dim)

# Crear los índices de posición para los embeddings posicionales
positions = tf.range(start=0, limit=num_patches, delta=1)

# Crear una capa de Embedding funcionalmente
positional_embeddings_layer = tf.keras.layers.Embedding(
    input_dim=num_patches,
    output_dim=embedding_dim,
)

# Aplicar la capa de Embedding a las posiciones
positional_embeddings = positional_embeddings_layer(positions)

# Expandir las dimensiones para que coincidan con el batch
positional_embeddings = tf.expand_dims(
    positional_embeddings, axis=0
)  # Shape: (1, num_patches, embedding_dim)

# Añadir los embeddings posicionales
encoded_patches = patch_embeddings + positional_embeddings

# Aplicar capas Transformer
num_layers = 4
num_heads = 8

# Ajustar la máscara de atención para MultiHeadAttention
# Crear la matriz de máscara para atención con la forma (batch_size, num_patches, num_patches)
attention_mask = tf.matmul(
    tf.expand_dims(mask_patches, -1), tf.expand_dims(mask_patches, 1)
)  # Shape: (batch_size, num_patches, num_patches)

# Convertir a enteros para ser utilizado en atención
attention_mask = tf.cast(attention_mask > 0, dtype=tf.int32)

for _ in range(num_layers):
    # Layer Normalization
    x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Multi-Head Attention
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embedding_dim // num_heads, dropout=0.1
    )(x1, x1, attention_mask=attention_mask)
    # Residual Connection
    x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
    # Layer Normalization y MLP
    x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = tf.keras.layers.Dense(embedding_dim * 4, activation=tf.keras.activations.gelu)(
        x3
    )
    x3 = tf.keras.layers.Dense(embedding_dim, activation=tf.keras.activations.gelu)(x3)
    # Residual Connection
    encoded_patches = tf.keras.layers.Add()([x3, x2])

# Representación final del encoder
representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
representation = tf.keras.layers.GlobalAvgPool1D()(representation)

# Representación codificada a partir de GlobalAvgPool1D (después del encoder)
latent_representation = representation  # Shape: (batch_size, embedding_dim)

# Proyectar la representación latente a un número mayor de características
expanded_representation = tf.keras.layers.Dense(
    units=num_patches * embedding_dim, activation="relu"
)(
    latent_representation
)  # Shape: (batch_size, num_patches * embedding_dim)

# Cambiar la forma de la representación para alinear con el volumen de parches
expanded_representation = tf.reshape(
    expanded_representation, (batch_size, num_patches, embedding_dim)
)  # Shape: (batch_size, num_patches, embedding_dim)

# Reconstruir los parches a su forma original con una capa densa
patch_dim = patch_size_t * patch_size_h * patch_size_w * C
decoded_patches = tf.keras.layers.Dense(patch_dim, activation="relu")(
    expanded_representation
)  # Shape: (batch_size, num_patches, patch_dim)

# Reconstruir el volumen de parches a la forma original usando reshape
decoded_patches = tf.reshape(
    decoded_patches,
    (
        batch_size,
        T // patch_size_t,
        patch_size_t,
        H // patch_size_h,
        patch_size_h,
        W // patch_size_w,
        patch_size_w,
        C,
    ),
)

# Reorganizar las dimensiones para reconstruir el volumen original
decoded_patches = tf.transpose(
    decoded_patches, perm=[0, 1, 3, 5, 2, 4, 6, 7]
)  # Shape: (batch_size, T, H, W, C)

# Ajustar a la forma original del video mediante un reshape final
reconstructed_video = tf.reshape(decoded_patches, [batch_size, T, H, W, C])

# Definir el modelo
autoencoder = tf.keras.Model(
    inputs=[input_data, attention_mask_input], outputs=reconstructed_video
)
encoder = tf.keras.Model(
    inputs=[input_data, attention_mask_input], outputs=representation
)


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
    mask = y_true[
        ..., 0:1
    ]  # Máscara de atención (Shape: batch_size, time_steps, H, W, 1)
    y_true = y_true[
        ..., 1:
    ]  # Características reales (Shape: batch_size, time_steps, H, W, C)

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


dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_types=(
        {"input_observations": tf.float32, "attention_mask": tf.int32},
        tf.float32,
    ),
    output_shapes=(
        {
            "input_observations": (T, H, W, C),
            "attention_mask": (T, H, W, C),
        },
        (T, H, W + 1, C),
    ),
)

sample = dataset.take(1)
for inputs, _ in sample:
    sample = inputs

batch_size = 4
buffer_size = 16

dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

total_epochs = len(transformed_dataset.index.get_level_values("epoch").unique())

# Compilar el modelo
autoencoder.compile(
    optimizer=tf.keras.optimizers.AdamW(),
    loss=masked_mse_loss,
)

# Expandir obs_min y obs_max para que tengan la forma adecuada para la operación de broadcasting
obs_min_exp = np.expand_dims(obs_min, axis=(0, 2))  # De (14,) a (1, 14, 1)
obs_max_exp = np.expand_dims(obs_max, axis=(0, 2))  # De (14,) a (1, 14, 1)


def visualizar_secuencia(
    input_obs, predicted_obs, mask, num_frames=100, save="autoencoder.gif"
):
    # Nos aseguramos de que la dimensión de batch sea 1 y luego la quitamos
    predicted_obs = np.squeeze(
        predicted_obs, axis=0
    )  # Shape resultante: (100, 285, 15, 1)

    # Quitamos la última dimensión (el canal) para tener las imágenes listas para mostrar
    input_obs = np.squeeze(input_obs, axis=-1)  # Shape resultante: (100, 285, 15)
    predicted_obs = np.squeeze(
        predicted_obs, axis=-1
    )  # Shape resultante: (100, 285, 15)
    mask = np.squeeze(mask, axis=-1)  # Shape resultante: (100, 285, 15)

    input_obs_shape = input_obs.shape
    predicted_obs_shape = predicted_obs.shape

    input_obs = input_obs.reshape(-1, input_obs_shape[-1])
    predicted_obs = predicted_obs.reshape(-1, predicted_obs_shape[-1])

    input_obs[..., 4:] = scaler.inverse_transform(input_obs[..., 4:])
    predicted_obs[..., 4:] = scaler.inverse_transform(predicted_obs[..., 4:])

    input_obs = input_obs.reshape(input_obs_shape)
    predicted_obs = predicted_obs.reshape(predicted_obs_shape)

    # Transponemos las imágenes para que el ancho sea 285 y el alto sea 15
    input_obs = np.transpose(input_obs, (0, 2, 1))  # Shape resultante: (100, 15, 285)
    predicted_obs = np.transpose(
        predicted_obs, (0, 2, 1)
    )  # Shape resultante: (100, 15, 285)
    mask_transposed = np.transpose(
        mask, (0, 2, 1)
    )  # Cambiando la forma a (100, 14, 286)

    # Calculamos la diferencia entre el Ground Truth y la Predicción
    diff_obs = np.abs(input_obs - predicted_obs)

    # Normalizamos las imágenes para que estén en el rango [0, 1] para visualizar correctamente
    input_obs = np.clip(
        (input_obs - obs_min_exp) / ((obs_max_exp - obs_min_exp) + 1e-9),
        a_min=0,
        a_max=1,
    )
    predicted_obs = np.clip(
        (predicted_obs - obs_min_exp) / ((obs_max_exp - obs_min_exp) + 1e-9),
        a_min=0,
        a_max=1,
    )
    diff_min_exp = np.expand_dims(np.min(diff_obs, axis=(0, 2)), axis=(0, 2))
    diff_max_exp = np.expand_dims(np.max(diff_obs, axis=(0, 2)), axis=(0, 2))
    diff_obs = np.clip(
        (diff_obs - diff_min_exp) / ((diff_max_exp - diff_min_exp) + 1e-9),
        a_min=0,
        a_max=1,
    )

    input_obs = input_obs * 255
    predicted_obs = predicted_obs * 255
    diff_obs = diff_obs * 255

    # Aplicar la máscara de atención
    input_obs = input_obs * mask_transposed
    predicted_obs = predicted_obs * mask_transposed
    diff_obs = diff_obs * mask_transposed

    row_labels = transformed_dataset.columns.to_list()

    # Configuramos la figura para la animación
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle("Ground Truth vs Prediction vs Difference")

    # Inicializamos los tres subplots
    ax_gt = axes[0]
    ax_pred = axes[1]
    ax_diff = axes[2]

    im_gt = ax_gt.imshow(
        input_obs[0],
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        aspect="auto",
    )
    im_pred = ax_pred.imshow(
        predicted_obs[0],
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        aspect="auto",
    )
    im_diff = ax_diff.imshow(
        diff_obs[0],
        cmap="hot",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        aspect="auto",
    )

    # Añadir cuadrícula para cada uno de los subplots
    for ax in axes:
        ax.grid(True, which="both", color="black", linestyle="-", linewidth=0.5)
        ax.set_yticks(np.arange(-0.5, input_obs.shape[1], 1), minor=True)

    # Añadir nombres de las filas en el eje Y
    ax_gt.set_yticks(np.arange(len(row_labels)))
    ax_gt.set_yticklabels(row_labels, fontsize=12)
    ax_pred.set_yticks(np.arange(len(row_labels)))
    ax_pred.set_yticklabels(row_labels, fontsize=12)
    ax_diff.set_yticks(np.arange(len(row_labels)))
    ax_diff.set_yticklabels(row_labels, fontsize=12)

    ax_gt.set_title("Ground Truth")
    ax_gt.axis("on")
    ax_pred.set_title("Prediction")
    ax_pred.axis("on")
    ax_diff.set_title("Difference")
    ax_diff.axis("on")

    # Función de actualización para la animación
    def actualizar(frame):
        im_gt.set_array(input_obs[frame])
        im_pred.set_array(predicted_obs[frame])
        im_diff.set_array(diff_obs[frame])
        return im_gt, im_pred, im_diff

    # Crear la animación
    anim = FuncAnimation(fig, actualizar, frames=num_frames, interval=200, blit=False)

    # Guardar la animación como GIF
    gif_path = os.path.join("reports/", save)
    anim.save(gif_path, writer="pillow", fps=5)

    # Subir el GIF a Neptune.ai
    run[f"visualizations/{save}"].upload(gif_path)

    plt.close(fig)


ground_truth = sample["input_observations"]
mask = sample["attention_mask"]

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
visualizar_secuencia(
    ground_truth.numpy(), predicted_obs, mask.numpy(), save="pretrain.gif"
)


# Definir el número de épocas y pasos por época
num_epochs = 100
steps_per_epoch = total_epochs // time_steps

# Entrenar el modelo
autoencoder.fit(
    dataset.repeat(steps_per_epoch * num_epochs),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
)

autoencoder.save("models/autoencoder.keras")
run["autoencoder"].upload("models/autoencoder.keras")
visualizar_secuencia(
    ground_truth.numpy(), predicted_obs, mask.numpy(), save="trained.gif"
)

Logger.reset()
run.stop()
