import collections
import tensorflow as tf


DTYPE = tf.float32
FORMAT = "NHWC"
global_variables = collections.defaultdict(list)


# Define basic operations ---
def he_init_w(shape: list, name: str = None, model_name: str = None) -> tf.Variable:
    fan_in = tf.cast(tf.reduce_prod(shape[:-1]), dtype=DTYPE)
    std = tf.sqrt(tf.cast(2, dtype=DTYPE)) / tf.sqrt(fan_in)
    v = tf.Variable(
        initial_value=tf.random.normal(shape=shape, mean=0, stddev=std, dtype=DTYPE),
        shape=shape,
        name=name,
    )

    # For tf2 track variables manually
    if model_name:
        global_variables[model_name].append(v)

    return v


def zero_init_w(shape: list, name: str = "None", model_name: str = None) -> tf.Variable:
    v = tf.Variable(
        initial_value=tf.zeros(shape=shape, dtype=DTYPE), shape=shape, name=name
    )

    # For tf2 track variables manually
    if model_name:
        global_variables[model_name].append(v)

    return v


def pixel_norm(x: tf.Tensor, epsilon: float = 1e-8) -> tf.Tensor:
    return x * tf.math.rsqrt(
        tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon
    )


def lerp(x0: tf.Tensor, x1: tf.Tensor, alpha: float) -> tf.Tensor:
    assert 0 <= alpha <= 1
    return tf.add(tf.multiply(x0, alpha), tf.multiply(x1, (1 - alpha)))


def dense(
    fan_in: int,
    fan_out: int,
    add_bias: bool = True,
    model_name: str = None,
    block_name: str = "",
):
    w = he_init_w(
        shape=[fan_in, fan_out], name=f"w_dense_{block_name}", model_name=model_name
    )
    if add_bias:
        b = zero_init_w(
            shape=[fan_out], name=f"b_dense_{block_name}", model_name=model_name
        )

        def process(x: tf.Tensor) -> tf.Tensor:
            return tf.add(tf.matmul(x, w), b)

        return process

    def process(x: tf.Tensor) -> tf.Tensor:
        return tf.matmul(x, w)

    return process


def conv2d(
    fan_in: int,
    f_maps: int,
    kernel_hw_dims: tuple,
    add_bias: bool = True,
    model_name: str = None,
    block_name: str = "",
):
    assert len(kernel_hw_dims) == 2
    w = he_init_w(
        shape=[kernel_hw_dims[0], kernel_hw_dims[1], fan_in, f_maps],
        name=f"w_conv_{block_name}",
        model_name=model_name,
    )
    if add_bias:
        b = zero_init_w(
            shape=[1, 1, 1, f_maps], name=f"b_conv_{block_name}", model_name=model_name
        )

        def process(x: tf.Tensor) -> tf.Tensor:
            return tf.add(
                tf.nn.conv2d(
                    x, w, strides=[1, 1, 1, 1], padding="SAME", data_format=FORMAT
                ),
                b,
            )

        return process

    def process(x: tf.Tensor) -> tf.Tensor:
        return tf.nn.conv2d(
            x, w, strides=[1, 1, 1, 1], padding="SAME", data_format=FORMAT
        )

    return process


def resize2d(x: tf.Tensor, scale: float = 2.0) -> tf.Tensor:
    assert scale > 0
    if scale == 1:
        return x

    elif scale > 1:
        # Nearest Neighbour upsampling
        scale = int(scale)
        x_shape = tf.shape(x)
        # Given that data is NHWC, add dimensions after resize channels (H and W)
        x = tf.reshape(x, [-1, x_shape[1], 1, x_shape[2], 1, x_shape[3]])
        # Tile added dimensions by a scale factor to replicate H and W dims
        x = tf.tile(x, [1, 1, scale, 1, scale, 1])
        # Reshape back to original dims with H and W scaled
        x = tf.reshape(x, [-1, x_shape[1] * scale, x_shape[2] * scale, x_shape[3]])
        return x

    elif 0 < scale < 1:
        # Average pooling
        scale = int(1 / scale)
        return tf.nn.avg_pool(
            x,
            ksize=[1, scale, scale, 1],
            strides=[1, scale, scale, 1],
            padding="VALID",
            data_format=FORMAT,
        )


def std_dev_layer(x: tf.Tensor, group_size: int = 4):
    group_size = tf.minimum(group_size, tf.shape(x)[0])
    x_dim = tf.shape(x)
    # [GMHWC] Split minibatch into M groups of size G
    y = tf.reshape(x, [group_size, -1, x_dim[1], x_dim[2], x_dim[3]])
    # [GMHWC] Subtract mean over group
    y -= tf.reduce_mean(y, axis=0, keepdims=True)
    # [MHWC]  Calc variance over group
    y = tf.reduce_mean(tf.square(y), axis=0)
    # [MHWC]  Calc stddev over group
    y = tf.sqrt(y + 1e-8)
    # [M111]  Take average over fmaps and pixels
    y = tf.reduce_mean(y, axis=[3, 1, 2], keepdims=True)
    # [NHW1]  Replicate over group and pixels
    y = tf.tile(y, [group_size, x_dim[1], x_dim[2], 1])
    return tf.concat([x, y], axis=3)


# Define Models blocks ---
def g_base_block(
    latent_dim: int, fan_out: int, model_name: str = "G", block_name: str = "base_block"
):
    dense_node = dense(
        fan_in=latent_dim,
        fan_out=fan_out * 16,
        add_bias=True,
        model_name=model_name,
        block_name=block_name,
    )
    conv2d_node = conv2d(
        fan_in=fan_out,
        f_maps=fan_out,
        kernel_hw_dims=(3, 3),
        add_bias=True,
        model_name=model_name,
        block_name=block_name,
    )

    def process(x: tf.Tensor) -> tf.Tensor:
        # Dense
        z = dense_node(pixel_norm(x))
        z = tf.reshape(z, shape=[-1, 4, 4, fan_out])
        z = tf.nn.leaky_relu(z, alpha=0.2)
        z = pixel_norm(z)

        # Conv
        z = conv2d_node(z)
        z = tf.nn.leaky_relu(z, alpha=0.2)
        z = pixel_norm(z)
        return z

    return process


def d_base_block(
    fan_in: int,
    fan_out_conv: int,
    fan_mid: int,
    label_dim: int = 0,
    group_dim: int = 4,
    model_name: str = "D",
    block_name: str = "base_block",
):
    conv_node = conv2d(
        fan_in=fan_in + 1,
        f_maps=fan_out_conv,
        kernel_hw_dims=(3, 3),
        add_bias=True,
        model_name=model_name,
        block_name=block_name,
    )
    dense_node = dense(
        fan_in=16 * fan_out_conv,
        fan_out=fan_mid,
        add_bias=True,
        model_name="D",
        block_name=block_name,
    )
    critic_node = dense(
        fan_in=fan_mid,
        fan_out=1 + label_dim,
        add_bias=True,
        model_name="D",
        block_name=block_name,
    )

    def process(x: tf.Tensor) -> tf.Tensor:
        # Conv
        z = std_dev_layer(x, group_dim)
        z = conv_node(z)
        z = tf.nn.leaky_relu(z, alpha=0.2)

        # Critic
        z = tf.reshape(z, shape=[-1, 16 * fan_out_conv])
        z = dense_node(z)
        z = tf.nn.leaky_relu(z, alpha=0.2)
        z = critic_node(z)
        return z

    return process


def g_block(
    fan_in: int, fan_out: int, model_name: str = "G", block_name: str = "block"
):
    conv2d_node_0 = conv2d(
        fan_in=fan_in,
        f_maps=fan_out,
        kernel_hw_dims=(3, 3),
        add_bias=True,
        model_name=model_name,
        block_name=block_name,
    )
    conv2d_node_1 = conv2d(
        fan_in=fan_out,
        f_maps=fan_out,
        kernel_hw_dims=(3, 3),
        add_bias=True,
        model_name=model_name,
        block_name=block_name,
    )

    def process(x: tf.Tensor) -> tf.Tensor:
        z = resize2d(x, scale=2.0)

        # Conv 0
        z = conv2d_node_0(z)
        z = pixel_norm(z)

        # Conv 1
        z = conv2d_node_1(z)
        z = pixel_norm(z)
        return z

    return process


def d_block(
    fan_in: int,
    fan_mid: int,
    fan_out: int,
    model_name: str = "D",
    block_name: str = "block",
):
    conv2d_node_0 = conv2d(
        fan_in=fan_in,
        f_maps=fan_mid,
        kernel_hw_dims=(3, 3),
        add_bias=True,
        model_name=model_name,
        block_name=block_name,
    )
    conv2d_node_1 = conv2d(
        fan_in=fan_mid,
        f_maps=fan_out,
        kernel_hw_dims=(3, 3),
        add_bias=True,
        model_name=model_name,
        block_name=block_name,
    )

    def process(x: tf.Tensor) -> tf.Tensor:
        # Conv 0
        z = conv2d_node_0(x)
        z = tf.nn.leaky_relu(z, alpha=0.2)

        # Conv 1
        z = conv2d_node_1(z)
        z = tf.nn.leaky_relu(z, alpha=0.2)

        # Downsample
        z = resize2d(z, scale=0.5)
        return z

    return process


def to_rgb(fan_in: int, num_channels: int, model_name: str = "G", block_name: str = "to_rgb"):
    conv2d_node = conv2d(
        fan_in=fan_in,
        f_maps=num_channels,
        kernel_hw_dims=(1, 1),
        add_bias=True,
        model_name=model_name,
        block_name=block_name,
    )

    def process(x: tf.Tensor) -> tf.Tensor:
        return conv2d_node(x)

    return process


def from_rgb(fan_out: int, num_channels: int, model_name: str = "D", block_name: str = "from_rgb"):
    conv2d_node = conv2d(
        fan_in=num_channels,
        f_maps=fan_out,
        kernel_hw_dims=(1, 1),
        add_bias=True,
        model_name=model_name,
        block_name=block_name,
    )

    def process(x: tf.Tensor) -> tf.Tensor:
        return conv2d_node(x)

    return process


# Define networks ---
class ProgG:
    def __init__(self, latent_dim: int, num_channels: int, max_fmaps: int, base_fmaps: int):
        # Dimensions
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.max_fmaps = max_fmaps
        self.base_fmaps = base_fmaps

        # Growing parameters
        self.curr_stage = 1
        self.alpha = 0
        self.new_block = None
        self.old_out = None
        self.new_out = to_rgb(fan_in=self.nf(1), num_channels=self.num_channels, block_name=f"to_rgb_{self.curr_stage}")

        # Final model blocks
        self.blocks = [g_base_block(latent_dim=self.latent_dim, fan_out=self.nf(1))]

    def nf(self, stage):
        return min(int(self.base_fmaps / (2.0 ** (stage * 1.0))), self.max_fmaps)

    def grow(self):
        """
        Adds new blocks to a model, while maintaining old ones
        for a later smooth transition using linear interpolation.
        """
        self.curr_stage += 1
        self.alpha = 0
        next_nf = self.nf(self.curr_stage)
        last_nf = self.nf(self.curr_stage - 1)

        # Append last block (if exists)
        if self.new_block:
            self.blocks.append(self.new_block)

        # Create new block and update to_rgb outs
        self.new_block = g_block(
            fan_in=last_nf, fan_out=next_nf, block_name=f"block_{self.curr_stage}"
        )
        self.old_out = self.new_out
        self.new_out = to_rgb(fan_in=next_nf, num_channels=self.num_channels, block_name=f"to_rgb_{self.curr_stage}")

        # Remove redundant weights from weight update
        vars_to_remove = [
            f"w_conv_to_rgb_{self.curr_stage-2}:0",
            f"b_conv_to_rgb_{self.curr_stage-2}:0",
        ]
        global_variables["G"] = [
            v for v in global_variables["G"] if v.name not in vars_to_remove
        ]

    def forward(self, x):
        """
        Processes x tensor, while TRAINING the Generator model.
        """
        for block in self.blocks:
            x = block(x)

        if self.curr_stage == 1:
            return self.new_out(x)

        y_new = self.new_out(self.new_block(x))
        y_old = resize2d(self.old_out(x), scale=2.0)
        return lerp(y_new, y_old, self.alpha)

    def set_alpha(self, val: float):
        """
        Sets the alpha parameter for a linear interpolation
        between new and old block.
        """
        assert 0 <= val <= 1
        self.alpha = val

    def get_final_model(self):
        """
        Returns a final model, converted to tf.keras.Model format.
        """
        model_in = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = model_in

        for block in self.blocks:
            x = block(x)

        if self.new_block:
            x = self.new_block(x)

        model_out = self.new_out(x)
        return tf.keras.models.Model(
            inputs=[model_in], outputs=[model_out], name="ProgG"
        )


class ProgD:
    def __init__(self, num_channels: int, max_fmaps: int, base_fmaps: int, label_dim: int = 0, group_dim: int = 4):
        # Dimensions
        self.group_dim = group_dim
        self.num_channels = num_channels
        self.max_fmaps = max_fmaps
        self.base_fmaps = base_fmaps

        # Growing parameters
        self.curr_stage = 1
        self.alpha = 0
        self.new_block = None
        self.old_in = None
        self.new_in = from_rgb(
            fan_out=self.nf(2), num_channels=self.num_channels, block_name=f"from_rgb_{self.curr_stage}"
        )

        # Final model blocks
        self.blocks = [
            d_base_block(
                fan_in=self.nf(2),
                fan_out_conv=self.nf(1),
                fan_mid=self.nf(0),
                label_dim=label_dim,
                group_dim=group_dim,
            )
        ]

    def nf(self, stage):
        return min(int(self.base_fmaps / (2.0 ** (stage * 1.0))), self.max_fmaps)

    def grow(self):
        self.curr_stage += 1
        self.alpha = 0
        next_nf = self.nf(self.curr_stage)
        last_nf = self.nf(self.curr_stage - 1)

        # Insert first block (if exists)
        if self.new_block:
            self.blocks.insert(0, self.new_block)

        # Create new block and update from_rgb outs inputs
        self.old_in = self.new_in
        self.new_in = from_rgb(
            fan_out=next_nf, num_channels=self.num_channels, block_name=f"from_rgb_{self.curr_stage}"
        )
        self.new_block = d_block(
            fan_in=next_nf,
            fan_mid=last_nf,
            fan_out=last_nf,
            block_name=f"block_{self.curr_stage}",
        )

        # Remove redundant weights from weight update
        vars_to_remove = [
            f"w_conv_from_rgb_{self.curr_stage-2}:0",
            f"b_conv_from_rgb_{self.curr_stage-2}:0",
        ]
        global_variables["D"] = [
            v for v in global_variables["D"] if v.name not in vars_to_remove
        ]

    def forward(self, x):
        if self.curr_stage == 1:
            return self.blocks[-1](self.new_in(x))

        x_old = self.old_in(resize2d(x, scale=0.5))
        x_new = self.new_block(self.new_in(x))

        x_comb = lerp(x_new, x_old, self.alpha)
        for block in self.blocks:
            x_comb = block(x_comb)

        return x_comb

    def set_alpha(self, val: float):
        """
        Sets the alpha parameter for a linear interpolation
        between new and old block.
        """
        assert 0 <= val <= 1
        self.alpha = val

    def get_final_model(self, input_dim: tuple):
        model_in = tf.keras.layers.Input(shape=input_dim)
        x = self.new_in(model_in)
        if self.new_block:
            x = self.new_block(x)

        for block in self.blocks:
            x = block(x)

        return tf.keras.models.Model(inputs=[model_in], outputs=[x], name="ProgD")
