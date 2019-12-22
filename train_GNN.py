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
import numpy as np
from pathlib import Path

from feature_stack_layer import FeatureStack
from GIN_on_stack import GINonStack

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# # Load datum

dataset = 'AIDS'
top_dir = Path('/local/scratch/ssd2/jkahn')
dataset_dir = top_dir / dataset

feat_X = np.load(dataset_dir / 'feat_X.npy')
adj_X = np.load(dataset_dir / 'adj_X.npy')
y = np.load(dataset_dir / 'y.npy')

# ### Splitty brah

(
    feat_X_train, feat_X_test,
    adj_X_train, adj_X_test,
    y_train, y_test
) = train_test_split(
    feat_X, adj_X, y,
    train_size=0.9,
)

feat_X_train.shape

# ## Set up network

num_nodes = feat_X.shape[1]
num_feats = feat_X.shape[-1]

backend.clear_session()


# +
def make_prediction_model(l2_strength=1e-5, emb_size=16):
    
    adj_input = layers.Input(shape=(num_nodes, num_nodes), name='adj_input')
    feat_input = layers.Input(shape=(num_nodes, num_feats), name='feat_input')
#     pdg_input = layers.Input(shape=(num_nodes,), name='pdg_input')
#     pdg_l = layers.Embedding(
#         input_dim=len(pdgTokens),
#         output_dim=emb_size,
#         name='embedding',
#     )(pdg_input)

    l = FeatureStack()([feat_input, adj_input])
    l = layers.Conv2D(8, (1, 1), use_bias=False)(l)
    l = layers.Conv2D(8, (1, 1), use_bias=False)(l)
    l = GINonStack(8, use_bias=False)(l)
    
    l = FeatureStack()([l, adj_input])
    l = layers.Conv2D(8, (1, 1), use_bias=False)(l)
    l = layers.Conv2D(8, (1, 1), use_bias=False)(l)
    l = GINonStack(8, use_bias=False)(l)
    
    l = layers.GlobalAveragePooling1D()(l)
    #l = layers.LeakyReLU()(layers.Dense(32, kernel_regularizer=regularizers.l2(l2_strength))(l))

    output_layer = layers.Dense(1, activation='sigmoid')(l)
    
    model = tf.keras.Model([adj_input, feat_input], output_layer)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()
    return model

pred_model = make_prediction_model()
# -

pred_model.fit(
    x={
        'adj_input': adj_X_train,
        'feat_input': feat_X_train,
    },
    y=y_train,
    batch_size=32,
    epochs=100,
    validation_data=(
        {
            'adj_input': adj_X_test,
            'feat_input': feat_X_test, 
        },
        y_test,
    )
)
