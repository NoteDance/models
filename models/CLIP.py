import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2d,BatchNormalization,LayerNormalization,MultiHeadAttention
from tensorflow.keras.layers import ZeroPadding2D,AveragePooling2D,Identity
from tensorflow.keras import Model
import numpy as np
from typing import Tuple, Union


class Bottleneck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(planes, 1, use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu1 = tf.nn.relu

        self.zeropadding2d = ZeroPadding2D(padding=1)
        self.conv2 = Conv2d(planes, 3, use_bias=False)
        self.bn2 = BatchNormalization()
        self.relu2 = tf.nn.relu

        self.avgpool = AveragePooling2D(stride, stride, 'VALID') if stride > 1 else Identity()

        self.conv3 = Conv2d(planes * self.expansion, 1, use_bias=False)
        self.bn3 = BatchNormalization()
        self.relu3 = tf.nn.relu

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = tf.keras.Sequential()
            self.downsample.add(AveragePooling2D(stride, stride, 'VALID'))
            self.downsample.add(Conv2d(planes * self.expansion, 1, strides=1, use_bias=False))
            self.downsample.add(BatchNormalization())

    def __call__(self, x):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.zeropadding2d(out)
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d:
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        self.positional_embedding = tf.Variable(tf.random.normal([spacial_dim ** 2 + 1, embed_dim]) / embed_dim ** 0.5)
        self.k_proj = Dense(embed_dim)
        self.q_proj = Dense(embed_dim)
        self.v_proj = Dense(embed_dim)
        self.c_proj = Dense(output_dim or embed_dim)
        self.num_heads = num_heads

    def __call__(self, x):
        shape = x.shape
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        new_shape = (batch_size, height * width, channels)
        x = tf.transpose(tf.reshape(x, new_shape), (1, 0, 2))
        x = tf.concat([tf.reduce_mean(x, axis=0, keepdims=True), x], axis=0)  # (HW+1)NC
        x = x + tf.cast(self.positional_embedding[:, None, :], x.dtype)  # (HW+1)NC
        tgt_len, bsz, embed_dim = x.shape
        query=self.q_proj(x[:1])
        key=self.k_proj(x)
        value=self.v_proj(x)
        query = tf.reshape(query, [bsz, 1, self.num_heads, -1])
        query = tf.transpose(query, [0, 2, 1, 3])
        query = tf.multiply(query, 1.0 / tf.math.sqrt(float(embed_dim)))
        key = tf.reshape(key, [bsz, tgt_len, self.num_heads, -1])
        key = tf.transpose(key, [0, 2, 3, 1])
        value = tf.reshape(value, [bsz, tgt_len, self.num_heads, -1])
        value = tf.transpose(value, [0, 2, 1, 3])
        qk = tf.matmul(query, key)
        w = tf.nn.softmax(qk)
        wv = tf.reshape(tf.transpose(tf.matmul(w, value), [0, 2, 1, 3]), [1, bsz, -1])
        x = self.c_proj(wv)
        return tf.squeeze(x, 0)


class ModifiedResNet:
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.zeropadding2d = ZeroPadding2D(padding=1)
        self.conv1 = Conv2d(width // 2, kernel_size=3, strides=2, use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu1 = tf.nn.relu
        self.conv2 = Conv2d(width // 2, kernel_size=3, use_bias=False)
        self.bn2 = BatchNormalization()
        self.relu2 = tf.nn.relu
        self.conv3 = Conv2d(width, kernel_size=3, use_bias=False)
        self.bn3 = BatchNormalization()
        self.relu3 = tf.nn.relu
        self.avgpool = AveragePooling2D(2, 2, 'VALID')

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = tf.keras.Sequential()
        layers.add(Bottleneck(self._inplanes, planes, stride))

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.add(Bottleneck(self._inplanes, planes))

        return layers

    def __call__(self, x):
        def stem(x):
            x = self.zeropadding2d(x)
            x = self.conv1(x)
            x = self.relu1(self.bn1(x))
            x = self.zeropadding2d(x)
            x = self.conv2(x)
            x = self.relu2(self.bn2(x))
            x = self.zeropadding2d(x)
            x = self.conv3(x)
            x = self.relu3(self.bn3(x))
            x = self.avgpool(x)
            return x

        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm:
    """Subclass torch's LayerNorm to handle fp16."""
    def __init__(self, input_size):
        self.layer_norm = LayerNormalization()

    def __call__(self, x):
        orig_type = x.dtype
        ret = self.layer_norm(tf.cast(x, tf.float32))
        return tf.cast(ret, orig_type)


class QuickGELU(tf.keras.layers.Layer):
    def __init__(self):
        super(QuickGELU, self).__init__()
        
    def __call__(self, x):
        return x * tf.nn.sigmoid(1.702 * x)


class ResidualAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, n_head: int, attn_mask = None):
        super(ResidualAttentionBlock, self).__init__()
        self.attn = MultiHeadAttention(n_head, d_model)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = tf.keras.Sequential()
        self.mlp.add(Dense(d_model * 4))
        self.mlp.add(QuickGELU())
        self.mlp.add(Dense(d_model))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = tf.cast(self.attn_mask, x.dtype) if self.attn_mask is not None else None
        return self.attn(x, mask=self.attn_mask)[0]

    def __call__(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer:
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None):
        self.width = width
        self.layers = layers
        self.resblocks = tf.keras.Sequential()
        for _ in range(layers):
            self.resblocks.add(ResidualAttentionBlock(width, heads, attn_mask))

    def __call__(self, x):
        return self.resblocks(x)


class VisionTransformer:
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = Conv2d(width, kernel_size=patch_size, strides=patch_size, use_bias=False)

        scale = width ** -0.5
        self.class_embedding = tf.Variable(scale * tf.random.normal([width]))
        self.positional_embedding = tf.Variable(scale * tf.random.normal((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = tf.Variable(scale * tf.random.normal(width, output_dim))

    def __call__(self, x, train_flag=True):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = tf.reshape(x, [x.shape[0], x.shape[1], -1])  # shape = [*, width, grid ** 2]
        x = tf.transpose(x, (0, 2, 1))  # shape = [*, grid ** 2, width]
        x = tf.concat([tf.cast(self.class_embedding, x.dtype) + tf.zeros([x.shape[0], 1, x.shape[-1]], dtype=x.dtype), x], axis=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + tf.cast(self.positional_embedding, x.dtype)
        x = self.ln_pre(x)

        x = tf.transpose(x, (1, 0, 2))  # NLD -> LND
        x = self.transformer(x)
        x = tf.transpose(x, (1, 0, 2))  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = tf.matmul(x, self.proj)

        return x


class CLIP(Model):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super(CLIP, self).__init__()
        
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = tf.Variable(tf.random.normal((vocab_size, transformer_width), 
                                                stddev=0.02))
        self.positional_embedding = tf.Variable(tf.random.normal((self.context_length, transformer_width), 
                                                 stddev=0.01
                                                 ))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = tf.Variable(tf.random.normal((transformer_width, embed_dim), 
                                            stddev=self.transformer.width ** -0.5, 
                                            ))
        self.logit_scale = tf.Variable(tf.ones([]) * np.log(1 / 0.07))

    def build_attention_mask(self):
        mask = tf.ones((self.context_length, self.context_length))
        mask = tf.linalg.band_part(mask, 0, -1) # zero out the upper diagonal
        mask = mask * -1e9 # fill with -1e9
        return mask

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = tf.gather(self.token_embedding, text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = tf.transpose(x, (1, 0, 2))  # NLD -> LND
        x = self.transformer(x)
        x = tf.transpose(x, (1, 0, 2))  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = tf.matmul(tf.gather_nd(x, tf.stack([tf.range(x.shape[0], dtype='int32'), 
                        tf.argmax(text, axis=-1, output_type='int32')], axis=1)), self.text_projection)

        return x

    def __call__(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / tf.norm(image_features, axis=1, keepdims=True)
        text_features = text_features / tf.norm(text_features, axis=1, keepdims=True)

        # cosine similarity as logits
        logit_scale = tf.math.exp(self.logit_scale)
        logits_per_image = tf.matmul(logit_scale * image_features, tf.transpose(text_features))
        logits_per_text = tf.transpose(logits_per_image)

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text