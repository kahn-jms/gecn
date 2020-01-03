# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Train a graph net

# Choose a GPU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tensorflow as tf
from tensorflow.keras import regularizers, callbacks, optimizers, backend
try:
    from tensorflow.keras import layers, activations
except ModuleNotFoundError:
    from tensorflow.python.keras import layers, activations

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import numpy as np
from pathlib import Path
import datetime

from feature_stack_layer import FeatureStack
from GIN_on_stack import GINonStack

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# # Load datum

# dataset = 'AIDS'
dataset = 'PROTEINS_full' #'AIDS'
top_dir = Path('/local/scratch/ssd2/jkahn/benchmarks')
dataset_dir = top_dir / dataset

# +
feat_X = np.load(dataset_dir / 'feat_X.npy')
adj_X = np.load(dataset_dir / 'adj_X.npy')
y = np.load(dataset_dir / 'y.npy')

# Just for this dataset
if dataset == 'PROTEINS_full':
    y = y - 1
# -

feat_X.shape, adj_X.shape, y.shape

# ### Splitty brah

(
    feat_X_train, feat_X_test,
    adj_X_train, adj_X_test,
    y_train, y_test
) = train_test_split(
    feat_X, adj_X, y,
    train_size=0.9,
)

feat_X_train.shape, y_train[:,0].shape

class_weights = compute_class_weight('balanced', np.unique(y_train[:,0]), y_train[:,0])
class_weights

# ## Set up network

num_nodes = feat_X.shape[1]
num_feats = feat_X.shape[-1]

backend.clear_session()


# +
def make_prediction_model(l2_strength=1e-5):
    
    adj_input = layers.Input(shape=(num_nodes, num_nodes), name='adj_input')
    feat_input = layers.Input(shape=(num_nodes, num_feats), name='feat_input')
#     pdg_input = layers.Input(shape=(num_nodes,), name='pdg_input')
#     pdg_l = layers.Embedding(
#         input_dim=len(pdgTokens),
#         output_dim=emb_size,
#         name='embedding',
#     )(pdg_input)

    l = FeatureStack()([feat_input, adj_input])
#     l = layers.Conv2D(32, (1, 1), use_bias=True)(l)
#     l = layers.LeakyReLU()(l)
#     l = layers.Conv2D(32, (1, 1), use_bias=True)(l)
#     l = layers.LeakyReLU()(l)
    l = layers.Conv2D(64, (1, 1), use_bias=True)(l)
    l = layers.LeakyReLU()(l)
    l = GINonStack(64, use_bias=True)(l)
    l = layers.LeakyReLU()(l)
    
#     l = FeatureStack()([l, adj_input])
#     l = layers.Conv2D(32, (1, 1), use_bias=True)(l)
#     l = layers.LeakyReLU()(l)
#     l = layers.Conv2D(32, (1, 1), use_bias=True)(l)
#     l = layers.LeakyReLU()(l)
# #     l = layers.Conv2D(32, (1, 1), use_bias=True)(l)
# #     l = layers.LeakyReLU()(l)
#     l = GINonStack(32, use_bias=True)(l)
#     l = layers.LeakyReLU()(l)
        
#     l = layers.GlobalMaxPooling1D()(l)
    l = layers.Flatten()(l)
#     l = layers.LeakyReLU()(layers.Dense(32, kernel_regularizer=regularizers.l2(l2_strength))(l))
#     l = layers.BatchNormalization()(l)
#     l = layers.LeakyReLU()(layers.Dense(16, kernel_regularizer=regularizers.l2(l2_strength))(l))
#     l = layers.BatchNormalization()(l)
#     l = layers.Dense(8, kernel_regularizer=regularizers.l2(l2_strength))(l)
#     l = layers.LeakyReLU()(l)
#     l = layers.BatchNormalization()(l)
    
    output_layer = layers.Dense(1, activation='sigmoid', name='y')(l)
    
    model = tf.keras.Model([adj_input, feat_input], output_layer)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()
    return model

pred_model = make_prediction_model()

run = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
# os.makedirs('./logs/' + run + '/plugins/profile', exist_ok=True)
# os.makedirs(os.path.join('model_checkpoints/', skim_wg, run), exist_ok=True)
print('Run:', run)
# -

pred_model.fit(
    x={
        'adj_input': adj_X_train[:2000],
        'feat_input': feat_X_train[:2000],
    },
    y=y_train[:2000],
    class_weight={'y': class_weights},
    batch_size=3,
    epochs=20,
    validation_data=(
        {
            'adj_input': adj_X_test[:200],
            'feat_input': feat_X_test[:200], 
        },
        y_test[:200],
    ),
    callbacks=[
        callbacks.ReduceLROnPlateau(factor=0.5, verbose=1, patience=5),
        callbacks.EarlyStopping(monitor='val_loss', patience=11, min_delta=1e-3),
#         callbacks.ModelCheckpoint(
#             os.path.join('./model_checkpoints', skim_wg, run, 'saved_model.h5'),
#             monitor='val_loss',
#             verbose=1,
#             save_best_only=True,
#         ),
#         callbacks.CSVLogger(
#             os.path.join('./model_checkpoints', skim_wg, run, 'training_history.csv'),
#             append=True,
#         )
        callbacks.TensorBoard('./logs/' + run, histogram_freq=1, write_grads=True, write_images=False)
    ]
)
