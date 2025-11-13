from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import KFold
from Model import *
from feature_encoding import *
from utils import *
from tensorflow.keras.models import load_model
import os
import shutil
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import random

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
tf.random.set_seed(1)  # for reproducibility
random.seed(0)

# 导入自定义层

if __name__ == '__main__':


    # Read the training set
    # train_pos_seqs = np.array(read_fasta('../data/strong.txt'))
    # train_neg_seqs = np.array(read_fasta('../data/weak.txt'))
    train_pos_seqs = np.array(read_fasta('../data/promoter.txt'))
    train_neg_seqs = np.array(read_fasta('../data/non-promoter.txt'))

    train_seqs = np.concatenate((train_pos_seqs, train_neg_seqs), axis=0)

    train_C2 = np.array(to_one_hot(train_seqs)).astype(np.float32)
    train_properties_code = np.array(to_properties_code(train_seqs)).astype(np.float32)

    train = np.concatenate((train_C2, train_properties_code), axis=1)
    # train = tf.reshape(train, (3382, 81, 7))
    train = tf.reshape(train, (6764, 81, 7))
    print(train.shape)  # (6764, 81, 7)

    # train_label = np.array([1] * 1591 + [0] * 1791).astype(np.float32)
    train_label = np.array([1] * 3382 + [0] * 3382).astype(np.float32)
    train_label = to_categorical(train_label, num_classes=2)

    BATCH_SIZE = 30
    EPOCHS = 200

    # Cross-validation
    n = 5
    k_fold = KFold(n_splits=n, shuffle=True, random_state=42)

    ten_all_performance = []
    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))

    # 创建模型保存根目录
    model_save_root = './saved_models'
    if not os.path.exists(model_save_root):
        os.makedirs(model_save_root)

    # Cycle 10 times (reduced to 2 for testing)
    for k in range(10):
        print('*' * 30 + ' the ' + str(k + 1) + ' cycle ' + '*' * 30)

        model = build_model(shape1=(81, 7), weight_decay=0.0001)
        model.summary()
        all_performance = []
        all_fprs = []
        all_tprs = []
        mean_fpr = np.linspace(0, 1, 100)

        # 为每个循环创建唯一的保存目录
        cycle_save_dir = os.path.join(model_save_root, f'cycle_{k + 1}')
        if not os.path.exists(cycle_save_dir):
            os.makedirs(cycle_save_dir)


        # 自定义回调函数，使用SavedModel格式保存并确保删除旧模型
        class CustomModelCheckpoint(tf.keras.callbacks.Callback):
            def __init__(self, save_dir):
                super().__init__()
                self.save_dir = save_dir
                self.best_val_loss = float('inf')

            def on_epoch_end(self, epoch, logs=None):
                current_val_loss = logs.get('val_loss')
                if current_val_loss < self.best_val_loss:
                    # 删除旧模型目录（如果存在）
                    if os.path.exists(self.save_dir):
                        shutil.rmtree(self.save_dir, ignore_errors=True)

                    self.best_val_loss = current_val_loss
                    # 使用SavedModel格式保存（推荐用于TensorFlow 2.4）
                    self.model.save(self.save_dir, save_format='tf')


        # 为每个fold创建唯一的保存目录
        all_checkpoint_dirs = []
        for fold_count in range(n):
            fold_checkpoint_dir = os.path.join(cycle_save_dir, f'fold_{fold_count + 1}')
            all_checkpoint_dirs.append(fold_checkpoint_dir)

        fold_histories = []
        ####
        

        # 5-fold cross-validations
        for fold_count, (train_index, val_index) in enumerate(k_fold.split(train)):
            print('*' * 30 + ' the ' + str(fold_count + 1) + ' fold ' + '*' * 30)

            train_index_tf = tf.convert_to_tensor(train_index, dtype=tf.int32)
            val_index_tf = tf.convert_to_tensor(val_index, dtype=tf.int32)
            trains = tf.gather(train, train_index_tf)
            val = tf.gather(train, val_index_tf)
            trains_label = tf.gather(train_label, train_index_tf)
            val_label = tf.gather(train_label, val_index_tf)

         fold_weight_path = os.path.join(cycle_save_dir, f'fold_{fold_count + 1}_best_model.h5')
            # Keras官方ModelCheckpoint，保存h5格式权重
            model_ckpt_callback = ModelCheckpoint(
                fold_weight_path,
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                save_weights_only=True,   # 只保存权重
                mode='min'
            )
                        
            ####(新加的history)
            history=model.fit(
                x=trains, y=trains_label, validation_data=(val, val_label), epochs=EPOCHS,
                batch_size=BATCH_SIZE, shuffle=True,
                callbacks=[EarlyStopping(monitor='val_loss', patience=22, mode='auto'),
                           model_ckpt_callback],
                verbose=1
            )
      fold_histories.append(history.history)
            ####


            val_pred = model.predict(val, verbose=1)

            # 计算性能指标
            Sn, Sp, Acc, MCC = show_performance(val_label[:, 1], val_pred[:, 1])
            AUC = roc_auc_score(val_label[:, 1], val_pred[:, 1])

            print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f' % (Sn, Sp, Acc, MCC, AUC))
            performance = [Sn, Sp, Acc, MCC, AUC]
            all_performance.append(performance)


        # 保存性能数据
        all_performance = np.array(all_performance)
        all_mean_performance = np.mean(all_performance, axis=0)
        ten_all_performance.append(all_mean_performance)


    # 显示最终结果
    print('---------------------------------------------5-cycle-result---------------------------------------')
    print(np.array(ten_all_performance))
    print('---------------------------------------------5-cycle-mean-result---------------------------------------')
    performance_mean = np.mean(np.array(ten_all_performance), axis=0)
    performance_std = np.std(np.array(ten_all_performance), axis=0)
    print(f'Mean Sn={performance_mean[0]:.4f}±{performance_std[0]:.4f}, '
      f'Sp={performance_mean[1]:.4f}±{performance_std[1]:.4f}, '
      f'Acc={performance_mean[2]:.4f}±{performance_std[2]:.4f}, '
      f'MCC={performance_mean[3]:.4f}±{performance_std[3]:.4f}, '
      f'AUC={performance_mean[4]:.4f}±{performance_std[4]:.4f}')
