import tensorflow as tf

from rlnav.types import vivit_utils as vutils


class ViViT_SpatioTemporal:
    def __init__(
        self,
        tubelet_size=(10, 286, 14),
        embedding_dim=128,
        num_layers=4,
        num_heads=8,
        window_size=100,
        height=286,
        width=14,
        channels=1,
    ):
        assert (
            window_size % tubelet_size[0] == 0
        ), f"The temporary tubelet size ({tubelet_size[0]}) must be divisor of the temporary window ({window_size})."
        assert (
            height % tubelet_size[1] == 0
        ), f"The size in height of the tubelet ({tubelet_size[1]}) must be divisor of the total height ({height})."
        assert (
            width % tubelet_size[2] == 0
        ), f"The size in width of the tubelet ({tubelet_size[2]}) must be divisor of the total width ({width})."

        # Parameters
        self.tubelet_size = tubelet_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.T = window_size
        self.H = height
        self.W = width
        self.C = channels

        # Models
        self.autoencoder = None
        self.encoder = None

    def build_model(self):
        # Inputs
        input_data = tf.keras.Input(
            shape=(self.T, self.H, self.W, self.C), name="input_observations"
        )
        attention_mask_input = tf.keras.Input(
            shape=(self.T, self.H, self.W, self.C), name="attention_mask"
        )

        input_shape = tf.shape(input_data)
        batch_size = input_shape[0]

        # Create tubelets (batch_size, num_tubelets, embedding_dim)
        num_tubelets, tubelets, mask_tubelets = vutils.create_tubelet_embedding(
            input_data,
            attention_mask_input,
            self.tubelet_size,
            self.embedding_dim,
        )

        positions = tf.range(start=0, limit=num_tubelets, delta=1)
        positional_embeddings_layer = tf.keras.layers.Embedding(
            input_dim=num_tubelets, output_dim=self.embedding_dim
        )
        positional_embeddings = positional_embeddings_layer(positions)
        positional_embeddings = tf.expand_dims(positional_embeddings, axis=0)
        encoded_tubelets = tubelets + positional_embeddings

        # Attention mask (batch_size, num_tubelets, num_tubelets)
        attention_mask = tf.matmul(
            tf.expand_dims(mask_tubelets, -1), tf.expand_dims(mask_tubelets, 1)
        )
        attention_mask = tf.cast(attention_mask > 0, dtype=tf.int32)

        # Apply Transformer
        for _ in range(self.num_layers):
            x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_tubelets)
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embedding_dim // self.num_heads,
                dropout=0.1,
            )(x1, x1, attention_mask=attention_mask)
            x2 = tf.keras.layers.Add()([attention_output, encoded_tubelets])
            x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = tf.keras.layers.Dense(
                self.embedding_dim * 4, activation=tf.keras.activations.gelu
            )(x3)
            x3 = tf.keras.layers.Dense(
                self.embedding_dim, activation=tf.keras.activations.gelu
            )(x3)
            encoded_tubelets = tf.keras.layers.Add()([x3, x2])

        # Encoder representation
        representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
            encoded_tubelets
        )
        latent_representation = tf.keras.layers.GlobalAvgPool1D()(representation)

        # Reconstruction
        expanded_representation = tf.keras.layers.Dense(
            units=num_tubelets * self.embedding_dim, activation="gelu"
        )(latent_representation)
        expanded_representation = tf.reshape(
            expanded_representation, (batch_size, num_tubelets, self.embedding_dim)
        )
        tubelet_dim = (
            self.tubelet_size[0] * self.tubelet_size[1] * self.tubelet_size[2] * self.C
        )
        decoded_tubelets = tf.keras.layers.Dense(tubelet_dim, activation="gelu")(
            expanded_representation
        )
        decoded_tubelets = tf.reshape(
            decoded_tubelets,
            (
                batch_size,
                self.T // self.tubelet_size[0],
                self.tubelet_size[0],
                self.H // self.tubelet_size[1],
                self.tubelet_size[1],
                self.W // self.tubelet_size[2],
                self.tubelet_size[2],
                self.C,
            ),
        )
        decoded_tubelets = tf.transpose(decoded_tubelets, perm=[0, 1, 3, 5, 2, 4, 6, 7])
        reconstructed_video = tf.reshape(
            decoded_tubelets, [batch_size, self.T, self.H, self.W, self.C]
        )

        # Define models
        self.autoencoder = tf.keras.Model(
            inputs=[input_data, attention_mask_input], outputs=reconstructed_video
        )
        self.encoder = tf.keras.Model(
            inputs=[input_data, attention_mask_input], outputs=latent_representation
        )

    def get_autoencoder(self):
        if self.autoencoder is None:
            raise ValueError("Autoencoder model has not been built.")
        return self.autoencoder

    def get_encoder(self):
        if self.encoder is None:
            raise ValueError("Encoder model has not been built.")
        return self.encoder


class ViViT_Factorised:
    def __init__(
        self,
        spatial_patch_size=(13, 14),
        embedding_dim=128,
        num_layers_spatial=4,
        num_layers_temporal=2,
        num_heads=8,
        window_size=100,
        height=286,
        width=14,
        channels=1,
        ff_dim=None,
    ):
        assert (
            height % spatial_patch_size[0] == 0
        ), f"The size in height of the patch ({spatial_patch_size[0]}) must be a divisor of the total height ({height})."
        assert (
            width % spatial_patch_size[1] == 0
        ), f"The size in width of the patch ({spatial_patch_size[1]}) must be a divisor of the total width ({width})."

        # Parámetros
        self.spatial_patch_size = spatial_patch_size
        self.embedding_dim = embedding_dim
        self.num_layers_spatial = num_layers_spatial
        self.num_layers_temporal = num_layers_temporal
        self.num_heads = num_heads
        self.T = window_size
        self.H = height
        self.W = width
        self.C = channels
        self.ff_dim = ff_dim if ff_dim is not None else embedding_dim * 4

        # Modelos
        self.autoencoder = None
        self.encoder = None

    def build_model(self):
        # Entradas
        input_data = tf.keras.Input(
            shape=(self.T, self.H, self.W, self.C), name="input_observations"
        )
        attention_mask_input = tf.keras.Input(
            shape=(self.T, self.H, self.W, self.C), name="attention_mask"
        )

        # Crear parches espaciales usando vutils.create_spatial_patches
        num_patches, spatial_patches, mask_patches = vutils.create_spatial_patches(
            input_data,
            attention_mask_input,
            self.spatial_patch_size,
            self.embedding_dim,
        )

        # Añadir embeddings posicionales espaciales
        spatial_patches = self.add_spatial_positional_embeddings(
            spatial_patches, num_patches
        )

        # Aplicar el encoder espacial usando TimeDistributed
        spatial_patches = self.apply_spatial_transformer(spatial_patches, mask_patches)

        # Obtener las representaciones por frame
        frame_representations = self.get_frame_representations(
            spatial_patches, mask_patches
        )

        # Añadir embeddings posicionales temporales
        frame_representations = self.add_temporal_positional_embeddings(
            frame_representations
        )

        # Aplicar el encoder temporal
        temporal_output = self.apply_temporal_transformer(frame_representations)

        # Obtener la representación latente
        latent_representation = tf.keras.layers.GlobalAveragePooling1D()(
            temporal_output
        )

        # Reconstrucción (opcional)
        reconstructed_video = self.build_decoder(latent_representation)

        # Definir los modelos
        self.autoencoder = tf.keras.Model(
            inputs=[input_data, attention_mask_input], outputs=reconstructed_video
        )
        self.encoder = tf.keras.Model(
            inputs=[input_data, attention_mask_input], outputs=latent_representation
        )

    def add_spatial_positional_embeddings(self, spatial_patches, num_patches):
        # Añadir embeddings posicionales espaciales
        position_indices = tf.range(num_patches)
        positional_embeddings_layer = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=self.embedding_dim
        )
        positional_embeddings = positional_embeddings_layer(position_indices)
        positional_embeddings = tf.expand_dims(
            tf.expand_dims(positional_embeddings, axis=0), axis=0
        )
        spatial_patches += positional_embeddings  # Broadcasting al batch_size y T
        return spatial_patches

    def apply_spatial_transformer(self, spatial_patches, mask_patches):
        # Definir el transformer espacial como un modelo
        def create_spatial_transformer():
            inputs = tf.keras.Input(shape=(None, self.embedding_dim))
            mask = tf.keras.Input(shape=(None,), dtype=tf.int32)

            x = inputs
            for _ in range(self.num_layers_spatial):
                # Layer Normalization
                x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
                # Multi-Head Attention con máscara
                attention_output = tf.keras.layers.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.embedding_dim // self.num_heads,
                    dropout=0.1,
                )(x1, x1, attention_mask=tf.expand_dims(mask, axis=1))
                # Conexión residual
                x2 = tf.keras.layers.Add()([attention_output, x])
                # MLP
                x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
                x3 = tf.keras.layers.Dense(self.ff_dim, activation="gelu")(x3)
                x3 = tf.keras.layers.Dense(self.embedding_dim)(x3)
                # Conexión residual
                x = tf.keras.layers.Add()([x3, x2])
            return tf.keras.Model(inputs=[inputs, mask], outputs=x)

        # Crear el modelo del transformer espacial
        spatial_transformer = create_spatial_transformer()

        # Aplicar TimeDistributed al transformer espacial
        spatial_patches = tf.keras.layers.TimeDistributed(spatial_transformer)(
            [spatial_patches, mask_patches]
        )

        return spatial_patches

    def get_frame_representations(self, spatial_patches, mask_patches):
        # Convertir la máscara a float
        mask = tf.cast(mask_patches, tf.float32)
        # Aplicar la máscara
        masked_patches = spatial_patches * tf.expand_dims(mask, axis=-1)
        # Sumar los parches válidos
        sum_patches = tf.reduce_sum(
            masked_patches, axis=2
        )  # (batch_size, T, embedding_dim)
        # Contar los parches válidos
        count_patches = tf.reduce_sum(mask, axis=2, keepdims=True)  # (batch_size, T, 1)
        # Evitar división por cero
        count_patches = tf.maximum(count_patches, 1.0)
        # Calcular el promedio
        frame_representations = (
            sum_patches / count_patches
        )  # (batch_size, T, embedding_dim)
        return frame_representations

    def add_temporal_positional_embeddings(self, frame_representations):
        T = self.T
        positions = tf.range(start=0, limit=T, delta=1)
        positional_embeddings_layer = tf.keras.layers.Embedding(
            input_dim=T, output_dim=self.embedding_dim
        )
        positional_embeddings = positional_embeddings_layer(positions)
        positional_embeddings = tf.expand_dims(
            positional_embeddings, axis=0
        )  # (1, T, embedding_dim)
        frame_representations += positional_embeddings  # Broadcasting al batch_size
        return frame_representations

    def apply_temporal_transformer(self, frame_representations):
        x = frame_representations
        for _ in range(self.num_layers_temporal):
            # Layer Normalization
            x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
            # Multi-Head Attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embedding_dim // self.num_heads,
                dropout=0.1,
            )(x1, x1)
            # Conexión residual
            x2 = tf.keras.layers.Add()([attention_output, x])
            # MLP
            x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = tf.keras.layers.Dense(self.ff_dim, activation="gelu")(x3)
            x3 = tf.keras.layers.Dense(self.embedding_dim)(x3)
            # Conexión residual
            x = tf.keras.layers.Add()([x3, x2])
        return x

    def build_decoder(self, latent_representation):
        def compute_kernel_and_stride(dim):
            """
            Computes the kernel size and stride for a dimension based on its prime factors.
            Args:
                dim (int): Dimension size (height or width).
            Returns:
                list: List of tuples (kernel_size, stride).
            """
            from sympy import factorint

            factors = list(factorint(dim).keys())  # Get prime factors
            strides = []
            kernel_sizes = []
            remaining_dim = dim

            for factor in factors:
                if remaining_dim % factor == 0:
                    strides.append(factor)
                    kernel_sizes.append(factor)
                    remaining_dim //= factor

            return list(zip(kernel_sizes, strides))

        # Reconstrucción del video a partir de la representación latente
        batch_size = tf.shape(latent_representation)[0]
        T = self.T
        H = self.H
        W = self.W
        C = self.C
        embedding_dim = self.embedding_dim

        # Expandimos la representación latente para incluir la dimensión temporal
        x = tf.expand_dims(
            latent_representation, axis=1
        )  # (batch_size, 1, embedding_dim)
        x = tf.tile(x, [1, T, 1])  # (batch_size, T, embedding_dim)

        # Proyectamos a initial_filters
        initial_filters = embedding_dim
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(initial_filters, activation="gelu")
        )(
            x
        )  # (batch_size, T, initial_filters)

        # Reestructuramos para aplicar Conv2DTranspose
        x = tf.reshape(x, (batch_size, T, 1, 1, initial_filters))

        current_height = 1
        current_width = 1
        current_filters = initial_filters

        height_steps = compute_kernel_and_stride(H // current_height)
        width_steps = compute_kernel_and_stride(W // current_width)

        for (h_kernel, h_stride), (w_kernel, w_stride) in zip(
            height_steps, width_steps
        ):
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2DTranspose(
                    filters=current_filters // 2,
                    kernel_size=(h_kernel, w_kernel),
                    strides=(h_stride, w_stride),
                    padding="valid",
                    activation="gelu",
                )
            )(x)
            current_filters //= 2
            current_height *= h_stride
            current_width *= w_stride

        # Reducir filtros progresivamente hasta alcanzar el número de canales objetivo
        while current_filters > C:
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=current_filters // 2,
                    kernel_size=(3, 3),
                    padding="same",
                    activation="gelu",
                )
            )(x)
            current_filters //= 2

        # Capa de salida
        reconstructed_video = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(
                filters=C,
                kernel_size=(3, 3),
                padding="same",
            )
        )(x)

        return reconstructed_video

    def get_autoencoder(self):
        if self.autoencoder is None:
            raise ValueError("El modelo autoencoder no ha sido construido.")
        return self.autoencoder

    def get_encoder(self):
        if self.encoder is None:
            raise ValueError("El modelo encoder no ha sido construido.")
        return self.encoder
