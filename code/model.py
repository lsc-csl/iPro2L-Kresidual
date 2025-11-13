from tensorflow.keras.layers import Input, Flatten, MaxPooling1D, AveragePooling1D,Bidirectional,GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Transformer_encoder import Transformer_encoder
from DenseNet import *
from tensorflow.keras.layers import Dense,LayerNormalization,LeakyReLU
from tensorflow.keras.regularizers import l2
from KanNet import *


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
