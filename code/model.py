from tensorflow.keras.layers import Input, Flatten, MaxPooling1D, AveragePooling1D,Bidirectional,GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from DenseNet import *
from tensorflow.keras.layers import Dense,LayerNormalization,LeakyReLU
from tensorflow.keras.regularizers import l2
from KanNet import *

import numpy as np
import tensorflow as tf


class positional(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        # 使用 **kwargs 来捕获额外的参数
        super(positional, self).__init__(**kwargs)
        self.d_model = d_model

    def call(self, sequences):
        max_sequence_len = sequences.shape[1]
        output = self.positional_encoding(max_sequence_len)
        return output

    def positional_encoding(self, max_len):
        pos = np.expand_dims(np.arange(0, max_len), axis=1)
        index = np.expand_dims(np.arange(0, self.d_model), axis=0)

        pe = self.angle(pos, index)

        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        pe = np.expand_dims(pe, axis=0)
        return tf.cast(pe, dtype=tf.float32)

    def angle(self, pos, index):
        return pos / np.power(10000, (index - index % 2) / np.float32(self.d_model))


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_h, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.d_h = d_h

    def call(self, query, key, value):
        matmul_q_and_transposed_k = tf.matmul(query, key, transpose_b=True)
        scale = tf.sqrt(tf.cast(self.d_h, dtype=tf.float32))
        scaled_attention_score = matmul_q_and_transposed_k / scale
        attention_weight = tf.nn.softmax(scaled_attention_score, axis=-1)

        return tf.matmul(attention_weight, value), attention_weight



class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, dropout_prob, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        # 确保 d_model 是整数，attention_head_count 是正整数
        if d_model % attention_head_count != 0:
            raise ValueError(
                f"d_model({d_model}) % attention_head_count({attention_head_count}) is not zero."
                f"d_model must be multiple of attention_head_count."
            )

        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.dropout_prob = dropout_prob

        # 强制确保分头后每个头的维度是整数
        self.d_h = int(d_model // attention_head_count)  # 确保是整数
        # self.Bigru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))
        # self.act = tf.keras.layers.Activation('tanh')
        # 使用 BiGRU 处理 Q、K 和 V
        self.gru_q = tf.keras.layers.GRU(d_model, return_sequences=True)
        self.gru_k = tf.keras.layers.GRU(d_model, return_sequences=True)
        self.gru_v = tf.keras.layers.GRU(d_model, return_sequences=True)

        # ScaledDotProductAttention 模块
        self.scaled_dot_product = ScaledDotProductAttention(self.d_h)

        # Conv1D 层
        self.conv1 = tf.keras.layers.Conv1D(filters=72, kernel_size=3, strides=1, padding='same', activation='relu',
                                            kernel_initializer='he_normal')
        self.conv2 = tf.keras.layers.Conv1D(filters=72, kernel_size=3, strides=1, padding='same', activation='relu',
                                            kernel_initializer='he_normal')
        # 合并层
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # 前馈网络层
        self.ff = tf.keras.layers.Dense(d_model)

    def split_head(self, tensor, batch_size):
        """将最后一个维度分成多个头"""
        return tf.transpose(
            tf.reshape(tensor, (batch_size, -1, self.attention_head_count, self.d_h)),
            [0, 2, 1, 3]
        )

    def concat_head(self, tensor, batch_size):
        """合并多个头"""
        return tf.reshape(
            tf.transpose(tensor, [0, 2, 1, 3]),
            (batch_size, -1, self.attention_head_count * self.d_h)
        )

    def call(self, query, key, value):
        batch_size = tf.shape(query)[0]

        # 使用 GRU 处理 Q、K 和 V
        query = self.gru_q(query)  # (batch_size, seq_len, d_model)
        key = self.gru_k(key)  # (batch_size, seq_len, d_model)
        value = self.gru_v(value)  # (batch_size, seq_len, d_model)

        # 分头处理
        query = self.split_head(query, batch_size)  # (batch_size, num_heads, seq_len, depth_per_head)
        key = self.split_head(key, batch_size)  # (batch_size, num_heads, seq_len, depth_per_head)
        value = self.split_head(value, batch_size)  # (batch_size, num_heads, seq_len, depth_per_head)

        # 计算注意力输出
        output, attention = self.scaled_dot_product(query, key, value)

        # 合并头
        output = self.concat_head(output, batch_size)  # (batch_size, seq_len, d_model)

        # 卷积层处理
        output1 = self.conv1(output)  # (batch_size, seq_len, 64)
        output2 = self.conv2(output)  # (batch_size, seq_len, 64)

        # 合并多种输出
        output5 = self.concat([output1, output2])  # (batch_size, seq_len, 256)
        # 前馈网络
        output5 = self.ff(output5)  # (batch_size, seq_len, d_model)

        return output5, attention


class PositionWiseFeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, d_point_wise_ff, d_model, **kwargs):
        super(PositionWiseFeedForwardLayer, self).__init__(**kwargs)
        self.w_1 = tf.keras.layers.Dense(d_point_wise_ff)
        self.w_2 = tf.keras.layers.Dense(d_model)

    def call(self, input):
        input = self.w_1(input)
        input = tf.nn.relu(input)
        return self.w_2(input)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.act = tf.keras.layers.Activation('relu')

        self.attention_head_count = attention_head_count



        self.ff = tf.keras.layers.Dense(d_model)

        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob

        self.multi_head_attention = MultiHeadAttention(attention_head_count, d_model, dropout_prob)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
            d_point_wise_ff,
            d_model
        )
        self.dropout_2 = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, input, training=True):
        attn_output, _ = self.multi_head_attention(input, input, input)
        attn_output = self.dropout_1(attn_output, training=training)
        out1 = self.layer_norm_1(tf.add(input, attn_output))  # residual network

        ffn_output = self.position_wise_feed_forward_layer(out1)
        ffn_output = self.dropout_2(ffn_output, training=training)
        out2 = self.layer_norm_2(tf.add(out1, ffn_output))  # residual network

        return out2



class Transformer_encoder(tf.keras.layers.Layer):
    def __init__(self, encoder_count, attention_head_count, d_model, d_point_wise_ff, dropout_prob, **kwargs):
        super(Transformer_encoder, self).__init__(**kwargs)

        # model hyper parameter variables
        self.encoder_count = encoder_count
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob

        self.pos_encoding = positional(d_model)
        self.encoder_embedding_dropout = tf.keras.layers.Dropout(dropout_prob)
        self.encoder_layers = [
            EncoderLayer(
                attention_head_count,
                d_model,
                d_point_wise_ff,
                dropout_prob
            ) for _ in range(encoder_count)
        ]

    def call(self,
             input,
             training=True
             ):

        encoder_tensor = input
        encoder_tensor = self.encoder_embedding_dropout(encoder_tensor, training=training)

        for i in range(self.encoder_count):
            encoder_tensor = self.encoder_layers[i](encoder_tensor, training=training)
        return encoder_tensor

    def get_config(self):
        config = super(Transformer_encoder, self).get_config()
        config.update({
            "encoder_count": self.encoder_count,
            "attention_head_count": self.attention_head_count,
            "d_model": self.d_model,
            "d_point_wise_ff": self.d_point_wise_ff,
            "dropout_prob": self.dropout_prob
        })
        return config


def label_smoothing_categorical_crossentropy(smoothing=0.2):
    def loss(y_true, y_pred):
        epsilon = 1e-8
        K = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true_smoothed = y_true * (1.0 - smoothing) + (smoothing / K)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
        return -tf.reduce_mean(tf.reduce_sum(y_true_smoothed * tf.math.log(y_pred), axis=1))
    return loss


def label_smoothing_penalty_loss(smoothing=0.1, penalty=2.0):
    def loss(y_true, y_pred):
        epsilon = 1e-8
        K = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true_smoothed = y_true * (1.0 - smoothing) + (smoothing / K)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
        ce = -y_true_smoothed * tf.math.log(y_pred)         # shape: [batch, 2]
        ce_sum = tf.reduce_sum(ce, axis=1)                  # shape: [batch]

        pred_class = tf.argmax(y_pred, axis=1)
        true_class = tf.argmax(y_true, axis=1)
        penal_mask = tf.cast((true_class == 1) & (pred_class == 0), tf.float32)
        penal_factor = 1.0 + (penalty - 1.0) * penal_mask  # shape: [batch]

        final_loss = ce_sum * penal_factor                  # [batch]
        return tf.reduce_mean(final_loss)
    return loss
# “置信度加权”+标签平滑
def label_smoothing_confidence_weighted_loss(smoothing=0.2, conf_threshold=0.95, conf_weight=0.5):
    def loss(y_true, y_pred):
        epsilon = 1e-8
        K = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true_smoothed = y_true * (1.0 - smoothing) + (smoothing / K)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
        ce = -tf.reduce_sum(y_true_smoothed * tf.math.log(y_pred), axis=1)
        conf = tf.reduce_max(y_pred, axis=1)
        weights = tf.where(conf > conf_threshold, conf_weight, 1.0)
        return tf.reduce_mean(ce * weights)
    return loss

import tensorflow as tf

def temp_scaled_label_smoothing_loss(smoothing=0.2, temperature=1.5):
    """
    温度标定标签平滑损失
    smoothing: 标签平滑参数
    temperature: softmax温度参数
    """
    def loss(y_true, y_pred):
        epsilon = 1e-8
        K = tf.cast(tf.shape(y_true)[-1], tf.float32)
        # 标签平滑
        y_true_smoothed = y_true * (1.0 - smoothing) + (smoothing / K)
        # 温度缩放softmax（如果y_pred已是概率，先还原logits再缩放）
        logits = tf.math.log(tf.clip_by_value(y_pred, epsilon, 1.0))
        logits_T = logits / temperature
        y_pred_temp = tf.nn.softmax(logits_T)
        # 交叉熵损失
        ce = -tf.reduce_sum(y_true_smoothed * tf.math.log(tf.clip_by_value(y_pred_temp, epsilon, 1.0)), axis=1)
        return tf.reduce_mean(ce)
    return loss

def temp_scaled_label_smoothing_reg_loss(smoothing=0.2, temperature=1.5, reg_lambda=0.01):
    """
    温度标定标签平滑损失 + 输出概率L2正则项
    smoothing: 标签平滑参数
    temperature: softmax温度参数
    reg_lambda: L2正则项系数
    """
    def loss(y_true, y_pred):
        epsilon = 1e-8
        K = tf.cast(tf.shape(y_true)[-1], tf.float32)
        # 标签平滑
        y_true_smoothed = y_true * (1.0 - smoothing) + (smoothing / K)
        # 温度缩放softmax（如果y_pred已是概率，先还原logits再缩放）
        logits = tf.math.log(tf.clip_by_value(y_pred, epsilon, 1.0))
        logits_T = logits / temperature
        y_pred_temp = tf.nn.softmax(logits_T)
        # 交叉熵损失
        ce = -tf.reduce_sum(y_true_smoothed * tf.math.log(tf.clip_by_value(y_pred_temp, epsilon, 1.0)), axis=1)
        loss_main = tf.reduce_mean(ce)
        # L2正则项
        reg_term = tf.reduce_mean(tf.reduce_sum(tf.square(y_pred), axis=1))
        # 总损失
        return loss_main + reg_lambda * reg_term
    return loss

def build_model(shape1, weight_decay):
    input = Input(shape=shape1)
    # 多尺度卷积特征提取
    conv1_3x3 = Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                       kernel_initializer='he_normal')(input)
    conv1_3x3 = MaxPooling1D(pool_size=3)(conv1_3x3)

    conv1_5x5 = Conv1D(filters=128, kernel_size=5, strides=1, padding='same', activation='relu',
                       kernel_initializer='he_normal')(input)
    conv1_5x5 = MaxPooling1D(pool_size=3)(conv1_5x5)

    conv1_7x7 = Conv1D(filters=128, kernel_size=7, strides=1, padding='same', activation='relu',
                       kernel_initializer='he_normal')(input)
    conv1_7x7 = MaxPooling1D(pool_size=3)(conv1_7x7)

    conv1_9x9 = Conv1D(filters=128, kernel_size=9, strides=1, padding='same', activation='relu',
                       kernel_initializer='he_normal')(input)
    conv1_9x9 = MaxPooling1D(pool_size=3)(conv1_9x9)

    # 特征融合
    multi_scale_features = Concatenate(axis=-1)([conv1_3x3, conv1_5x5, conv1_7x7, conv1_9x9])
    conv_output = BatchNormalization()(multi_scale_features)

    conv_output = Dropout(0.1)(conv_output)

    # 下采样卷积(原)
    conv = Conv1D(filters=128, kernel_size=3, strides=3, padding='valid', activation='relu',
                  kernel_initializer='he_normal')(conv_output)

    for i in range(4):
        if i == 0:
            kan_res_block = ResidualKANBlock(features_dim=128)
            res_out = kan_res_block(conv)
            res_out = Dropout(0.3)(res_out)
        else:
            res_out = ResidualKANBlock(features_dim=128)(res_out)
        
    res_out=LayerNormalization()(res_out)
    
    # Transformer编码器
    trans = Transformer_encoder(
        encoder_count=4,
        attention_head_count=4,
        d_model=128,
        d_point_wise_ff=64,
        dropout_prob=0.1
    )
    out = trans(res_out)
    encode_output = trans(out)
   
    lstm_1 = Bidirectional(GRU(units=64, name='BiGRU', return_sequences=True))(encode_output)

    out_1 = MaxPooling1D(pool_size=3, padding='same')(lstm_1)
    out_2 = AveragePooling1D(pool_size=3, padding='same')(lstm_1)
    out = Concatenate(axis=-1)([out_1, out_2])

    out = Flatten()(out)
    
    out = LayerNormalization()(out)

    out = Dense(units=32, activation='relu', kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay))(out)

    # MLP
    x = Dense(units=2, activation="softmax", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(out)

    inputs = [input]
    outputs = [x]

    model = Model(inputs=inputs, outputs=outputs, name="iPro2L-Kresidual")
    
    optimizer = Adam(lr=2.2e-4, epsilon=1e-8)

    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # model.compile(loss=temp_scaled_label_smoothing_loss(smoothing=0.1), optimizer=optimizer, metrics=['accuracy'])
    model.compile(loss=label_smoothing_categorical_crossentropy(smoothing=0.1), optimizer=optimizer, metrics=['accuracy'])
    # model.compile(loss=temp_scaled_label_smoothing_reg_loss(smoothing=0.2, temperature=1.5, reg_lambda=0), optimizer=optimizer, metrics=['accuracy'])

    return model




