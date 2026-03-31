# ============================================================
# KERAS MAIN CONCEPTS
# ============================================================
# Keras is a high-level deep learning API designed for fast
# experimentation. Keras 3 is backend-agnostic — it runs on
# TensorFlow, JAX, and PyTorch.
#
# Install: pip install keras tensorflow
# Backend: set via KERAS_BACKEND env var or ~/.keras/keras.json
#   "tensorflow" (default) | "jax" | "torch"
#
# import keras            # standalone Keras 3
# from tensorflow import keras  # bundled with TF (tf.keras)

import os
os.environ["KERAS_BACKEND"] = "tensorflow"   # set before importing keras

import keras
import numpy as np
import tensorflow as tf   # used for data pipelines in examples

print("Keras version:", keras.__version__)

# Synthetic datasets reused throughout
X_train = np.random.randn(800,  20).astype("float32")
y_train = np.random.randint(0, 5, 800)
X_val   = np.random.randn(200,  20).astype("float32")
y_val   = np.random.randint(0, 5, 200)

X_img_train = np.random.randn(200, 28, 28, 1).astype("float32")
y_img_train = np.random.randint(0, 10, 200)


# ============================================================
# 1. SEQUENTIAL API
# ============================================================
# The simplest way to build a model — a linear stack of layers.
# Use it when data flows straight through from input to output.

print("\n--- Sequential API ---")

model = keras.Sequential(
    [
        keras.layers.Input(shape=(20,)),          # explicit Input layer
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(5, activation="softmax"),
    ],
    name="sequential_mlp",
)
model.summary()

# Add layers incrementally
model2 = keras.Sequential(name="incremental")
model2.add(keras.layers.Dense(64, activation="relu", input_shape=(20,)))
model2.add(keras.layers.Dense(5,  activation="softmax"))

# Inspect layers
print("Layers:", [l.name for l in model.layers])
print("Output shape:", model.output_shape)


# ============================================================
# 2. FUNCTIONAL API
# ============================================================
# Builds a DAG of layers — supports multiple inputs/outputs,
# skip connections (ResNet-style), and shared layers.

print("\n--- Functional API ---")

# Single input / output
inputs  = keras.Input(shape=(20,), name="features")
x       = keras.layers.Dense(128, activation="relu")(inputs)
x       = keras.layers.BatchNormalization()(x)
x       = keras.layers.Dropout(0.3)(x)
x       = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(5, activation="softmax", name="predictions")(x)

func_model = keras.Model(inputs=inputs, outputs=outputs, name="func_mlp")
func_model.summary()

# Multiple inputs and outputs
input_a  = keras.Input(shape=(10,), name="input_a")
input_b  = keras.Input(shape=(10,), name="input_b")
merged   = keras.layers.Concatenate()([input_a, input_b])
shared   = keras.layers.Dense(64, activation="relu")(merged)
out_cls  = keras.layers.Dense(5,  activation="softmax", name="class_out")(shared)
out_reg  = keras.layers.Dense(1,  name="reg_out")(shared)

multi_model = keras.Model(
    inputs=[input_a, input_b],
    outputs={"class_out": out_cls, "reg_out": out_reg},
)
multi_model.summary()

# Residual / skip connection
def residual_block(x, units):
    shortcut = x
    x = keras.layers.Dense(units, activation="relu")(x)
    x = keras.layers.Dense(units)(x)
    x = keras.layers.Add()([x, shortcut])   # skip connection
    x = keras.layers.Activation("relu")(x)
    return x

inp = keras.Input(shape=(64,))
x   = residual_block(inp, 64)
x   = residual_block(x,  64)
out = keras.layers.Dense(5, activation="softmax")(x)
res_model = keras.Model(inp, out, name="residual")


# ============================================================
# 3. MODEL SUBCLASSING
# ============================================================
# Maximum flexibility — define forward pass in call().
# Required for dynamic architectures (variable-length loops, etc.)

print("\n--- Subclassing ---")

class MLP(keras.Model):
    def __init__(self, hidden_units, num_classes, dropout_rate=0.3):
        super().__init__()
        self.hidden_layers = [
            keras.layers.Dense(u, activation="relu")
            for u in hidden_units
        ]
        self.bn      = keras.layers.BatchNormalization()
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.output_layer = keras.layers.Dense(num_classes,
                                                activation="softmax")

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

mlp = MLP(hidden_units=[128, 64], num_classes=5)
print("Output:", mlp(X_train[:4], training=False).shape)

# Trainable / non-trainable params
mlp.build(input_shape=(None, 20))
print("Trainable params:", sum(np.prod(v.shape) for v in mlp.trainable_variables))


# ============================================================
# 4. LAYERS
# ============================================================

print("\n--- Layers ---")

# --- Core ---
keras.layers.Dense(64, activation="relu", use_bias=True,
                   kernel_regularizer=keras.regularizers.L2(1e-4))
keras.layers.Activation("relu")
keras.layers.Flatten()
keras.layers.Reshape((7, 7, 64))
keras.layers.Concatenate(axis=-1)
keras.layers.Add()
keras.layers.Multiply()

# --- Convolutional ---
keras.layers.Conv1D(32, kernel_size=3, strides=1, padding="same")
keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu")
keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same")   # upsampling
keras.layers.SeparableConv2D(64, 3, padding="same")              # depthwise
keras.layers.MaxPooling2D(pool_size=2)
keras.layers.AveragePooling2D(pool_size=2)
keras.layers.GlobalAveragePooling2D()
keras.layers.GlobalMaxPooling2D()

# --- Recurrent ---
keras.layers.LSTM(64, return_sequences=True,  return_state=True)
keras.layers.GRU(64,  return_sequences=False)
keras.layers.SimpleRNN(32)
keras.layers.Bidirectional(keras.layers.LSTM(64))
keras.layers.TimeDistributed(keras.layers.Dense(32))

# --- Normalisation ---
keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)
keras.layers.LayerNormalization()
keras.layers.GroupNormalization(groups=8)

# --- Regularisation ---
keras.layers.Dropout(rate=0.5)
keras.layers.SpatialDropout2D(rate=0.3)      # drops entire feature maps
keras.layers.GaussianNoise(stddev=0.1)       # data augmentation / regularisation

# --- Embedding ---
keras.layers.Embedding(input_dim=10000, output_dim=64, mask_zero=True)

# --- Attention ---
keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
keras.layers.Attention()

# --- Preprocessing (live inside the model) ---
keras.layers.Normalization()               # learned mean/variance normalisation
keras.layers.Rescaling(1.0 / 255.0)       # pixel scaling for images
keras.layers.RandomFlip("horizontal")      # data augmentation
keras.layers.RandomRotation(0.1)
keras.layers.RandomZoom(0.1)
keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=200)


# ============================================================
# 5. CUSTOM LAYERS
# ============================================================
# Subclass keras.Layer to create reusable, trainable building blocks.

print("\n--- Custom Layers ---")

class LinearWithBias(keras.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # build() is called once on first use — creates weights
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )
        super().build(input_shape)

    def call(self, x):
        return x @ self.w + self.b

    def get_config(self):              # enables model.save() to serialise
        config = super().get_config()
        config.update({"units": self.units})
        return config

layer = LinearWithBias(32)
out   = layer(tf.ones([4, 20]))
print("Custom layer output:", out.shape)

# Layer with non-trainable state
class RunningMean(keras.Layer):
    def build(self, input_shape):
        self.mean = self.add_weight(shape=(), initializer="zeros",
                                    trainable=False, name="running_mean")
        self.count = self.add_weight(shape=(), initializer="zeros",
                                     trainable=False, name="count")

    def call(self, x):
        batch_mean = tf.reduce_mean(x)
        self.count.assign_add(1.0)
        self.mean.assign(self.mean + (batch_mean - self.mean) / self.count)
        return x   # pass-through


# ============================================================
# 6. LOSSES & METRICS
# ============================================================

print("\n--- Losses & Metrics ---")

# Built-in losses
keras.losses.SparseCategoricalCrossentropy(from_logits=False)
keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
keras.losses.BinaryCrossentropy(from_logits=True)
keras.losses.MeanSquaredError()
keras.losses.MeanAbsoluteError()
keras.losses.Huber(delta=1.0)
keras.losses.KLDivergence()
keras.losses.CosineSimilarity()

# Built-in metrics
keras.metrics.SparseCategoricalAccuracy()
keras.metrics.CategoricalAccuracy()
keras.metrics.BinaryAccuracy()
keras.metrics.Precision()
keras.metrics.Recall()
keras.metrics.AUC(curve="ROC")
keras.metrics.MeanAbsoluteError()
keras.metrics.MeanSquaredError()

# Custom loss function
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred   = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    ce        = -y_true * tf.math.log(y_pred)
    weight    = alpha * y_true * (1 - y_pred) ** gamma
    return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))

# Custom metric (subclass)
class F1Score(keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall    = keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * p * r / (p + r + keras.backend.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


# ============================================================
# 7. OPTIMIZERS & LEARNING RATE SCHEDULES
# ============================================================

print("\n--- Optimizers ---")

keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999)
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
keras.optimizers.RMSprop(learning_rate=1e-3, rho=0.9)
keras.optimizers.Adagrad(learning_rate=0.01)
keras.optimizers.Adadelta(learning_rate=1.0)
keras.optimizers.Lion(learning_rate=1e-4)           # memory-efficient

# Learning rate schedules
cosine = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=1000, alpha=0.0,
)
exp_decay = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3, decay_steps=500, decay_rate=0.96,
)
poly = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1e-3, decay_steps=1000, end_learning_rate=1e-6,
)
piecewise = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[200, 500],
    values=[1e-3, 1e-4, 1e-5],
)

# Gradient clipping (prevents exploding gradients)
opt_clipped = keras.optimizers.Adam(learning_rate=1e-3,
                                     clipnorm=1.0)      # clip by norm
opt_clipped2 = keras.optimizers.Adam(learning_rate=1e-3,
                                      clipvalue=0.5)    # clip by value


# ============================================================
# 8. COMPILE & TRAIN
# ============================================================

print("\n--- Compile & Train ---")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    ],
)

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1,
)

# Results
print("History keys:", list(history.history.keys()))
print("Final val_accuracy:", history.history["val_accuracy"][-1])

# Evaluate
loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Eval — loss: {loss:.4f}  accuracy: {acc:.4f}")

# Predict
probs   = model.predict(X_val[:5])
classes = np.argmax(probs, axis=1)
print("Predicted classes:", classes)


# ============================================================
# 9. CALLBACKS
# ============================================================

print("\n--- Callbacks ---")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=0,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
    keras.callbacks.TensorBoard(
        log_dir="logs",
        histogram_freq=1,
    ),
    keras.callbacks.CSVLogger("training.csv", append=False),
    keras.callbacks.TerminateOnNaN(),           # stop if loss is NaN
    keras.callbacks.BackupAndRestore("backup"), # fault-tolerant training
]

# Custom callback — full hook list
class VerboseCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):   pass
    def on_epoch_begin(self, epoch, logs=None): pass

    def on_epoch_end(self, epoch, logs=None):
        lr = float(self.model.optimizer.learning_rate)
        print(f"\n[Epoch {epoch+1}] loss={logs['loss']:.4f}  lr={lr:.6f}")

    def on_batch_end(self, batch, logs=None):  pass
    def on_train_end(self, logs=None):         pass
    def on_test_begin(self, logs=None):        pass
    def on_predict_begin(self, logs=None):     pass


# ============================================================
# 10. REGULARIZATION TECHNIQUES
# ============================================================

print("\n--- Regularization ---")

# Weight regularization (L1, L2, L1L2)
reg_model = keras.Sequential([
    keras.layers.Dense(
        128, activation="relu", input_shape=(20,),
        kernel_regularizer=keras.regularizers.L2(1e-4),
        bias_regularizer=keras.regularizers.L1(1e-5),
    ),
    keras.layers.Dense(
        64, activation="relu",
        kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
    ),
    keras.layers.Dense(5, activation="softmax"),
])

# Dropout
drop_model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(20,)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(5, activation="softmax"),
])

# Batch Normalization (acts as light regularization too)
bn_model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(20,)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(5, activation="softmax"),
])

# Label smoothing (built into the loss)
smooth_loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# Activity regularization
act_reg = keras.layers.Dense(
    64,
    activation="relu",
    activity_regularizer=keras.regularizers.L2(1e-5),
)


# ============================================================
# 11. DATA PREPROCESSING LAYERS
# ============================================================
# Preprocessing layers live inside the model — no separate
# fit/transform step needed at inference time.

print("\n--- Preprocessing Layers ---")

# Numeric normalisation
normalizer = keras.layers.Normalization()
normalizer.adapt(X_train)                 # compute mean and variance

norm_model = keras.Sequential([
    normalizer,                            # applies (x - mean) / std
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(5, activation="softmax"),
])

# Image preprocessing pipeline
img_model = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),           # [0,255] -> [0,1]
    keras.layers.RandomFlip("horizontal"),          # augmentation
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.1),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])

# Text preprocessing
vectorizer = keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=200,
)
texts = ["hello world", "keras is great", "deep learning"]
vectorizer.adapt(texts)

text_model = keras.Sequential([
    vectorizer,
    keras.layers.Embedding(10000, 64, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(1, activation="sigmoid"),
])


# ============================================================
# 12. CNN — IMAGE CLASSIFICATION
# ============================================================

print("\n--- CNN ---")

cnn = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),

    # Block 1
    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2),

    # Block 2
    keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2),

    # Block 3
    keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),

    # Classifier
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax"),
], name="cnn")

cnn.compile(optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
cnn.summary()

dummy_img = np.random.randn(4, 28, 28, 1).astype("float32")
print("CNN output:", cnn.predict(dummy_img, verbose=0).shape)


# ============================================================
# 13. RECURRENT — LSTM TEXT CLASSIFICATION
# ============================================================

print("\n--- LSTM ---")

lstm_model = keras.Sequential([
    keras.layers.Input(shape=(50,)),                       # sequence of 50 token IDs
    keras.layers.Embedding(input_dim=5000, output_dim=64),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.LSTM(32),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation="sigmoid"),           # binary classification
], name="lstm_classifier")

lstm_model.compile(optimizer="adam",
                   loss="binary_crossentropy",
                   metrics=["accuracy"])
lstm_model.summary()

dummy_seq = np.random.randint(0, 5000, (8, 50))
print("LSTM output:", lstm_model.predict(dummy_seq, verbose=0).shape)


# ============================================================
# 14. TRANSFER LEARNING
# ============================================================

print("\n--- Transfer Learning ---")

# Load pretrained base (ImageNet weights, no top)
base = keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False             # freeze all base layers

# Build classification head
inputs  = keras.Input(shape=(96, 96, 3))
x       = keras.applications.mobilenet_v2.preprocess_input(inputs)
x       = base(x, training=False)
x       = keras.layers.GlobalAveragePooling2D()(x)
x       = keras.layers.Dense(256, activation="relu")(x)
x       = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)

tl_model = keras.Model(inputs, outputs, name="transfer_learning")
tl_model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

trainable = sum(1 for v in tl_model.trainable_variables)
total     = sum(1 for v in tl_model.variables)
print(f"Trainable vars: {trainable} / Total: {total}")

# Fine-tuning — unfreeze the last N layers of the base
base.trainable = True
fine_tune_from = len(base.layers) - 30
for layer in base.layers[:fine_tune_from]:
    layer.trainable = False

tl_model.compile(optimizer=keras.optimizers.Adam(1e-5),   # lower LR
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

# Available application models:
#   keras.applications.VGG16 / VGG19
#   keras.applications.ResNet50 / ResNet101 / ResNet152
#   keras.applications.MobileNetV2 / MobileNetV3Small/Large
#   keras.applications.EfficientNetV2B0..L
#   keras.applications.InceptionV3 / InceptionResNetV2
#   keras.applications.DenseNet121 / DenseNet201
#   keras.applications.Xception / NASNetMobile


# ============================================================
# 15. SAVING, LOADING & SERIALIZATION
# ============================================================

print("\n--- Save & Load ---")

# --- Keras native format (.keras) — recommended ---
model.save("model.keras")
loaded = keras.models.load_model("model.keras")
print("Loaded .keras model output:", loaded.predict(X_val[:2], verbose=0).shape)

# --- SavedModel format ---
model.save("saved_model_dir")
loaded2 = keras.models.load_model("saved_model_dir")

# --- Weights only ---
model.save_weights("weights.weights.h5")
model.load_weights("weights.weights.h5")

# --- Config + weights separately ---
config    = model.get_config()             # architecture as dict
rebuilt   = keras.Sequential.from_config(config)
rebuilt.set_weights(model.get_weights())   # transfer weights

# --- JSON serialization ---
json_str  = model.to_json()
from_json = keras.models.model_from_json(json_str)

# --- Custom objects (needed when loading custom layers/losses) ---
loaded_custom = keras.models.load_model(
    "model.keras",
    custom_objects={"LinearWithBias": LinearWithBias,
                    "focal_loss": focal_loss},
)

# Clean up
import shutil
for path in ["model.keras", "saved_model_dir", "weights.weights.h5",
             "best_model.keras", "training.csv", "logs", "backup"]:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)


# ============================================================
# 16. KERAS TUNER — HYPERPARAMETER TUNING
# ============================================================
# keras_tuner searches for the best hyperparameters automatically.
# Install: pip install keras-tuner

print("\n--- Keras Tuner ---")

# pip install keras-tuner
# import keras_tuner as kt

# def build_model(hp):
#     model = keras.Sequential()
#     model.add(keras.layers.Input(shape=(20,)))
#
#     # Tune number of layers
#     for i in range(hp.Int("num_layers", min_value=1, max_value=4)):
#         model.add(keras.layers.Dense(
#             units=hp.Int(f"units_{i}", min_value=32, max_value=256, step=32),
#             activation=hp.Choice("activation", ["relu", "gelu", "tanh"]),
#         ))
#         model.add(keras.layers.Dropout(
#             rate=hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.5, step=0.1),
#         ))
#
#     model.add(keras.layers.Dense(5, activation="softmax"))
#     model.compile(
#         optimizer=keras.optimizers.Adam(
#             hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
#         ),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )
#     return model
#
# # Tuner strategies
# tuner = kt.RandomSearch(build_model, objective="val_accuracy",
#                          max_trials=10, directory="tuner_dir")
#
# tuner = kt.BayesianOptimization(build_model, objective="val_accuracy",
#                                  max_trials=10, directory="tuner_dir")
#
# tuner = kt.Hyperband(build_model, objective="val_accuracy",
#                       max_epochs=20, directory="tuner_dir")
#
# # Run the search
# tuner.search(X_train, y_train, epochs=10,
#              validation_data=(X_val, y_val),
#              callbacks=[keras.callbacks.EarlyStopping(patience=2)])
#
# # Retrieve best model and hyperparameters
# best_hp    = tuner.get_best_hyperparameters(1)[0]
# best_model = tuner.get_best_models(1)[0]
# print("Best units:", best_hp.get("units_0"))
# print("Best lr:", best_hp.get("lr"))
# tuner.results_summary()

print("Keras Tuner example is commented out.")
print("Install with: pip install keras-tuner")


# ============================================================
# QUICK REFERENCE CHEAT SHEET
# ============================================================
#
# MODEL APIS
#   keras.Sequential([layers])               Linear stack
#   keras.Model(inputs, outputs)             Functional (DAG)
#   class MyModel(keras.Model)               Subclassing (dynamic)
#   model.summary()                          Print architecture
#
# COMMON LAYERS
#   Dense(units, activation, kernel_regularizer)  Fully connected
#   Conv2D(filters, kernel_size, padding)         Convolution
#   MaxPooling2D / GlobalAveragePooling2D         Pooling
#   LSTM / GRU / Bidirectional                    Recurrent
#   BatchNormalization / LayerNormalization        Normalization
#   Dropout(rate)                                 Regularization
#   Embedding(vocab_size, output_dim)             Token embedding
#   MultiHeadAttention(num_heads, key_dim)        Attention
#   Concatenate / Add / Multiply                  Merge
#   Normalization / Rescaling                     Preprocessing
#   RandomFlip / RandomRotation / RandomZoom      Augmentation
#   TextVectorization                             Text preprocessing
#
# CUSTOM LAYER
#   class MyLayer(keras.Layer):
#       def build(self, input_shape): ...    create weights
#       def call(self, x): ...              forward pass
#       def get_config(self): ...           for serialisation
#
# COMPILE
#   model.compile(optimizer, loss, metrics)
#   Losses:   SparseCategoricalCrossentropy / BinaryCrossentropy
#             MeanSquaredError / Huber / KLDivergence
#   Metrics:  SparseCategoricalAccuracy / Precision / Recall / AUC
#
# TRAIN
#   model.fit(X, y, epochs, batch_size,
#             validation_data, callbacks)
#   model.evaluate(X, y)
#   model.predict(X)
#
# CALLBACKS
#   EarlyStopping(monitor, patience, restore_best_weights)
#   ModelCheckpoint(filepath, save_best_only)
#   ReduceLROnPlateau(monitor, factor, patience)
#   TensorBoard(log_dir)
#   CSVLogger(filename)
#   TerminateOnNaN()
#
# REGULARIZATION
#   kernel_regularizer=keras.regularizers.L2(1e-4)
#   Dropout / SpatialDropout2D / GaussianNoise
#   BatchNormalization (implicit regularisation)
#   CategoricalCrossentropy(label_smoothing=0.1)
#   clipnorm / clipvalue in optimizer
#
# OPTIMIZERS
#   Adam / AdamW / SGD / RMSprop / Lion
#   Pass a schedule as learning_rate:
#     CosineDecay / ExponentialDecay / PolynomialDecay
#
# TRANSFER LEARNING
#   keras.applications.MobileNetV2(include_top=False, weights="imagenet")
#   base.trainable = False           freeze
#   layer.trainable = True           unfreeze selectively
#   Recompile after changing trainability
#
# SAVE & LOAD
#   model.save("model.keras")                  Recommended
#   keras.models.load_model("model.keras")     Load
#   model.save_weights("w.weights.h5")         Weights only
#   model.get_config() / from_config(config)   Architecture
#
# KERAS TUNER
#   kt.RandomSearch / BayesianOptimization / Hyperband
#   hp.Int / hp.Float / hp.Choice / hp.Boolean
#   tuner.search(X, y, ...) / tuner.get_best_models()
