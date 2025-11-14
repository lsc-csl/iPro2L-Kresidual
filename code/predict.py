from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from sklearn.model_selection import KFold
from Model import *          # 自定义模型
from feature_encoding import *
from utils import *          # 评估函数
import os
import random
import numpy as np
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE

seed = 1024
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# 参数配置
train_pos_file = '../data/promoter.txt'
train_neg_file = '../data/non-promoter.txt'
test_pos_file  = '../Liu_data/promoter.txt'
test_neg_file  = '../Liu_data/non-promoter.txt'

if __name__ == '__main__':
    train_pos_seqs = np.array(read_fasta(train_pos_file))
    train_neg_seqs = np.array(read_fasta(train_neg_file))
    train_seqs = np.concatenate((train_pos_seqs, train_neg_seqs), axis=0)
    train_C2 = np.array(to_one_hot(train_seqs)).astype(np.float32)
    train_properties_code = np.array(to_properties_code(train_seqs)).astype(np.float32)
    train = np.concatenate((train_C2, train_properties_code), axis=1)
    train = tf.reshape(train, (len(train_seqs), *shape))
    print("Train feature shape:", train.shape)

    train_label = np.array([1] * len(train_pos_seqs) + [0] * len(train_neg_seqs)).astype(np.float32)
    train_label = to_categorical(train_label, num_classes=2)

    # 直接加载模型
    best_model_dir = 'saved_models/cycle_1/fold_1'
    assert os.path.exists(best_model_dir), f"模型目录不存在: {best_model_dir}"
    model = tf.keras.models.load_model(best_model_dir)
    print(f"已加载完整模型: {best_model_dir}")

    # 预测与评估
    test_pred = model.predict(test_data, verbose=1)
    Sn, Sp, Acc, MCC = show_performance(test_label[:, 1], test_pred[:, 1])
    AUC = roc_auc_score(test_label[:, 1], test_pred[:, 1])
    print('Test Sn = %.6f, Sp = %.6f, Acc = %.6f, MCC = %.6f, AUC = %.6f' % (Sn, Sp, Acc, MCC, AUC))

    # 特征提取+tSNE
    feature_layer_name = 'transformer_encoder' 
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer(feature_layer_name).output)
    features = feature_extractor.predict(test_data)
    n, h, w = features.shape
    features_flat = features.reshape((n, h * w))
    tsne = TSNE(n_components=2, random_state=seed, init='random', learning_rate=350)
    tsne_results = tsne.fit_transform(features_flat)

    plt.figure(figsize=(8, 8))
    for label, color, name in zip([0, 1], ['#DC143C', '#800080'], ['Positive', 'Negative']):
        indices = (np.argmax(test_label, axis=1) == label)
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], c=color, label=name, alpha=0.5)
    plt.legend()
    plt.title('t-SNE of Model Features')
    plt.savefig('tsne_plot.png', format='png', dpi=300)
    plt.show()
