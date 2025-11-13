from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
class KANLinear(tf.keras.layers.Layer):

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, **kwargs):
        super(KANLinear, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_total = grid_size + 2 * spline_order + 1

        # Initialize grid
        h = 2.0 / grid_size
        grid = tf.range(-spline_order, grid_size + spline_order + 1, dtype=tf.float32) * h
        grid = tf.expand_dims(grid, 0)
        grid = tf.tile(grid, [in_features, 1])
        self.grid = tf.Variable(grid, trainable=False, name="grid")

        # Initialize weights
        self.base_weight = self.add_weight(
            shape=(in_features, out_features),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            name="base_weight"
        )
        self.spline_weight = self.add_weight(
            shape=(in_features, grid_size + spline_order, out_features),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            name="spline_weight"
        )

    def get_config(self):
        config = super(KANLinear, self).get_config()
        config.update({
            'in_features': self.in_features,
            'out_features': self.out_features,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order
        })
        return config

    def b_splines(self, x):

        x = tf.expand_dims(x, -1)
        grid = self.grid
        grid_total = self.grid_total

        # Initial bases (order 0)
        lower = grid[:, :-1]
        upper = grid[:, 1:]
        bases = tf.cast(
            tf.logical_and(x >= lower, x < upper),
            x.dtype
        )

        # Recursive basis calculation
        for k in range(1, self.spline_order + 1):
            # Compute denominators
            denom1 = grid[:, k:grid_total - 1] - grid[:, :grid_total - 1 - k]  # [in_features, grid_total-1-k]
            denom2 = grid[:, k + 1:grid_total] - grid[:, 1:grid_total - k]  # [in_features, grid_total-1-k]

            # Avoid division by zero
            denom1 = tf.where(tf.abs(denom1) < 1e-5, tf.ones_like(denom1), denom1)
            denom2 = tf.where(tf.abs(denom2) < 1e-5, tf.ones_like(denom2), denom2)

            # Compute terms
            term1 = (x - grid[:, :grid_total - 1 - k]) / denom1 * bases[...,
                                                                  :grid_total - 1 - k]  # [N, in_features, grid_total-1-k]
            term2 = (grid[:, k + 1:grid_total] - x) / denom2 * bases[...,
                                                               1:grid_total - k]  # [N, in_features, grid_total-1-k]

            bases = term1 + term2

        return bases  # [N, in_features, grid_size + spline_order]

    @tf.function
    def call(self, x, training=None):
        # Handle 2D/3D inputs
        input_shape = tf.shape(x)
        if len(x.shape) == 3:
            # Flatten batch and timesteps
            x_flat = tf.reshape(x, [-1, self.in_features])
        else:
            x_flat = x

        # Base linear transformation
        base_output_flat = tf.matmul(x_flat, self.base_weight)

        # Spline transformation
        spline_bases = self.b_splines(x_flat)  # [N_flat, in_features, grid_size+spline_order]
        spline_bases_flat = tf.reshape(
            spline_bases,
            [-1, self.in_features * (self.grid_size + self.spline_order)]
        )
        spline_output_flat = tf.matmul(
            spline_bases_flat,
            tf.reshape(self.spline_weight, [self.in_features * (self.grid_size + self.spline_order), self.out_features])
        )

        # Combine outputs
        output_flat = base_output_flat + spline_output_flat

        # Restore original dimensions if needed
        if len(x.shape) == 3:
            output = tf.reshape(output_flat, [input_shape[0], input_shape[1], self.out_features])
        else:
            output = output_flat

        return output

    def update_grid(self, x, margin=0.01):
        # Flatten if 3D
        if len(x.shape) == 3:
            x_flat = tf.reshape(x, [-1, self.in_features])
        else:
            x_flat = x

        # Update grid logic (unchanged)
        batch = tf.shape(x_flat)[0]
        x_sorted = tf.sort(x_flat, axis=0)
        indices = tf.cast(tf.linspace(0.0, tf.cast(batch - 1, tf.float32), self.grid_size + 1), tf.int32)
        grid_adaptive = tf.gather(x_sorted, indices, axis=0)

        x_min = tf.reduce_min(x_sorted, axis=0) - margin
        x_max = tf.reduce_max(x_sorted, axis=0) + margin
        uniform_step = (x_max - x_min) / self.grid_size
        grid_uniform = tf.expand_dims(x_min, 0) + tf.expand_dims(uniform_step, 0) * tf.range(self.grid_size + 1,
                                                                                             dtype=tf.float32)[:,
                                                                                    tf.newaxis]

        grid = 0.02 * grid_uniform + 0.98 * grid_adaptive
        grid_expanded = []
        for i in range(self.spline_order):
            grid_expanded.append(grid[0] - (i + 1) * uniform_step)
        grid_expanded.append(grid)
        for i in range(1, self.spline_order + 1):
            grid_expanded.append(grid[-1] + i * uniform_step)
        grid_expanded = tf.concat(grid_expanded, axis=0)
        grid_expanded = tf.transpose(grid_expanded)
        self.grid.assign(grid_expanded)


class ResidualKANBlock(tf.keras.layers.Layer):
    """Residual block with KANLinear layer (supports 3D inputs)"""

    def __init__(self, features_dim, grid_size=5, spline_order=3, strides=1, **kwargs):
        super(ResidualKANBlock, self).__init__(**kwargs)
        self.features_dim = features_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.strides = strides

        # KAN layer with convolution layers
        self.conv1 = Conv1D(
            filters=features_dim, kernel_size=3, padding='same', use_bias=False, strides=strides
        )
        self.bn1 = BatchNormalization()
        self.kan = KANLinear(
            in_features=features_dim,
            out_features=features_dim,
            grid_size=grid_size,
            spline_order=spline_order
        )
        self.conv2 = Conv1D(
            filters=features_dim, kernel_size=3, padding='same', use_bias=False
        )
        self.bn2 = BatchNormalization()

        # Shortcut connection
        self.shortcut = tf.keras.Sequential()
        # if strides > 1:
        self.shortcut.add(Conv1D(
            filters=features_dim, kernel_size=1,
            strides=strides, padding='same', use_bias=False
        ))
        self.shortcut.add(BatchNormalization())

    def get_config(self):
        config = super(ResidualKANBlock, self).get_config()
        config.update({
            'features_dim': self.features_dim,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
            'strides': self.strides
        })
        return config

    @tf.function
    def call(self, x, update_grid=False, training=None):
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        # KAN layer handles 3D internally
        x = self.kan(x, training=training)
        x = tf.nn.relu(x)

        x = tf.keras.layers.add([x, residual])
        x = tf.nn.relu(x)

        if update_grid:
            self.kan.update_grid(x)

        return x
