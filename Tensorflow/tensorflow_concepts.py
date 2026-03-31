# ============================================================
# TENSORFLOW MAIN CONCEPTS
# ============================================================
# TensorFlow is an end-to-end machine learning platform.
# Keras (tf.keras) is its high-level API for building and
# training models. TF 2.x runs eagerly by default — operations
# execute immediately, like NumPy.

import tensorflow as tf
import numpy as np
import os

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices("GPU"))


# ============================================================
# 1. TENSORS
# ============================================================
# Tensors are immutable, typed, n-dimensional arrays.
# Similar to NumPy arrays but can live on GPU.

print("\n--- Tensors ---")

# Creating tensors
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

print(tf.zeros([3, 4]))
print(tf.ones([2, 3]))
print(tf.fill([2, 2], 7.0))
print(tf.range(0, 10, 2))
print(tf.linspace(0.0, 1.0, 5))
print(tf.random.normal([3, 3]))          # standard normal
print(tf.random.uniform([3, 3]))         # uniform [0, 1)
print(tf.eye(3))                         # identity matrix

# Tensor attributes
t = tf.random.normal([4, 5])
print("shape :", t.shape)                # TensorShape([4, 5])
print("dtype :", t.dtype)               # tf.float32
print("ndim  :", t.ndim)
print("rank  :", tf.rank(t).numpy())

# Casting
x = tf.constant([1, 2, 3])
x = tf.cast(x, tf.float32)

# Convert to/from NumPy
arr = t.numpy()                          # tensor -> numpy
t2  = tf.constant(arr)                  # numpy -> tensor


# ============================================================
# 2. TENSOR OPERATIONS
# ============================================================

print("\n--- Tensor Operations ---")

a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

print(a + b)                             # element-wise add
print(tf.add(a, b))                      # same
print(a * b)                             # element-wise multiply
print(a @ b)                             # matrix multiply
print(tf.matmul(a, b))                   # same
print(tf.transpose(a))
print(tf.reduce_sum(a))                  # 10.0
print(tf.reduce_sum(a, axis=0))          # column sums
print(tf.reduce_mean(a))
print(tf.reduce_max(a), tf.reduce_min(a))
print(tf.argmax(a, axis=1))             # index of max per row
print(tf.square(a))
print(tf.sqrt(a))
print(tf.abs(tf.constant([-1.0, 2.0, -3.0])))

# Reshaping
t = tf.range(12, dtype=tf.float32)
print(tf.reshape(t, [3, 4]))
print(tf.reshape(t, [2, 2, 3]))

# Squeeze / expand dims
t = tf.random.normal([3, 1, 4])
print(tf.squeeze(t).shape)               # (3, 4)
print(tf.expand_dims(t, axis=0).shape)   # (1, 3, 1, 4)

# Concatenate / stack
a = tf.ones([3])
b = tf.zeros([3])
print(tf.concat([a, b], axis=0))         # [1 1 1 0 0 0]
print(tf.stack([a, b], axis=0))          # shape (2, 3)


# ============================================================
# 3. VARIABLES
# ============================================================
# tf.Variable is a mutable tensor — used for model weights.
# Unlike tf.constant, variables can be updated in place.

print("\n--- Variables ---")

v = tf.Variable([1.0, 2.0, 3.0])
print(v)
v.assign([4.0, 5.0, 6.0])               # replace all values
v.assign_add([1.0, 1.0, 1.0])           # v += 1
v.assign_sub([0.5, 0.5, 0.5])           # v -= 0.5

# Slice assignment
v[0].assign(99.0)

# Variables are automatically tracked by Keras layers and optimizers
w = tf.Variable(tf.random.normal([3, 3]), name="weights", trainable=True)
b = tf.Variable(tf.zeros([3]),            name="bias",    trainable=True)


# ============================================================
# 4. AUTOMATIC DIFFERENTIATION — GradientTape
# ============================================================
# tf.GradientTape records operations for automatic differentiation.

print("\n--- GradientTape ---")

x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y = x ** 3 + 3 * x               # y = x³ + 3x
dy_dx = tape.gradient(y, x)
print("dy/dx at x=2:", dy_dx.numpy()) # 3x² + 3 = 15

# Multiple variables
w = tf.Variable(tf.random.normal([2, 1]))
b = tf.Variable(tf.zeros([1]))
X = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y_true = tf.constant([[1.0], [0.0]])

with tf.GradientTape() as tape:
    y_pred = X @ w + b
    loss   = tf.reduce_mean((y_true - y_pred) ** 2)

grads = tape.gradient(loss, [w, b])
print("grad w:", grads[0].numpy())
print("grad b:", grads[1].numpy())

# Persistent tape (compute gradient multiple times)
with tf.GradientTape(persistent=True) as tape:
    z = x ** 2
dz_dx  = tape.gradient(z, x)
d2z_dx = tape.gradient(dz_dx, x)      # second derivative
del tape


# ============================================================
# 5. BUILDING MODELS WITH KERAS
# ============================================================

print("\n--- Keras Models ---")

# --- Sequential API (simple stacks) ---
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(20,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),  # multi-class output
])
model.summary()

# --- Functional API (complex topologies: multiple inputs/outputs, skip connections) ---
inputs  = tf.keras.Input(shape=(20,), name="features")
x       = tf.keras.layers.Dense(128, activation="relu")(inputs)
x       = tf.keras.layers.BatchNormalization()(x)
x       = tf.keras.layers.Dropout(0.3)(x)
x       = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

func_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mlp")
func_model.summary()

# --- Subclassing API (maximum flexibility) ---
class MyModel(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1     = tf.keras.layers.Dense(128, activation="relu")
        self.bn      = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.fc2     = tf.keras.layers.Dense(64, activation="relu")
        self.out     = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.bn(x, training=training)    # training flag matters for BN/Dropout
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return self.out(x)

custom_model = MyModel(num_classes=10)
dummy = tf.random.normal([4, 20])
print("Custom model output shape:", custom_model(dummy).shape)


# ============================================================
# 6. COMMON LAYERS
# ============================================================

print("\n--- Common Layers ---")

# Dense (fully connected)
fc = tf.keras.layers.Dense(64, activation="relu", use_bias=True)

# Convolutional
conv2d = tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                                 strides=1, padding="same", activation="relu")
pool   = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
gap    = tf.keras.layers.GlobalAveragePooling2D()

# Recurrent
lstm   = tf.keras.layers.LSTM(64, return_sequences=True)
gru    = tf.keras.layers.GRU(64, return_sequences=False)
bidir  = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))

# Normalization
bn     = tf.keras.layers.BatchNormalization()
ln     = tf.keras.layers.LayerNormalization()

# Regularization
drop   = tf.keras.layers.Dropout(rate=0.5)
l2_reg = tf.keras.regularizers.L2(1e-4)

# Embedding
emb    = tf.keras.layers.Embedding(input_dim=10000, output_dim=64)

# Reshape / utility
flat   = tf.keras.layers.Flatten()
reshape = tf.keras.layers.Reshape((7, 7, 64))
concat = tf.keras.layers.Concatenate()
add    = tf.keras.layers.Add()


# ============================================================
# 7. COMPILING MODELS
# ============================================================
# compile() sets the optimizer, loss, and metrics.

print("\n--- Compile ---")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
    ],
)

# Common losses:
#   SparseCategoricalCrossentropy   multi-class, integer labels
#   CategoricalCrossentropy         multi-class, one-hot labels
#   BinaryCrossentropy              binary classification
#   MeanSquaredError                regression
#   MeanAbsoluteError               regression, robust to outliers
#   Huber                           regression, robust to outliers

# Common optimizers:
#   Adam / AdamW / SGD / RMSprop / Adagrad / Adadelta

# Common metrics:
#   Accuracy / BinaryAccuracy / SparseCategoricalAccuracy
#   Precision / Recall / AUC / MeanAbsoluteError


# ============================================================
# 8. TRAINING WITH fit()
# ============================================================

print("\n--- Training ---")

# Generate synthetic data
X_train = np.random.randn(1000, 20).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)
X_val   = np.random.randn(200,  20).astype(np.float32)
y_val   = np.random.randint(0, 10, 200)

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1,
)

# Training history
print("Keys:", list(history.history.keys()))
print("Val accuracy:", history.history["val_accuracy"])

# Evaluate and predict
loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Val loss: {loss:.4f}  Val accuracy: {acc:.4f}")

probs   = model.predict(X_val[:5])        # probabilities
classes = np.argmax(probs, axis=1)        # predicted class
print("Predicted classes:", classes)


# ============================================================
# 9. CALLBACKS
# ============================================================
# Callbacks hook into the training loop to add behaviour.

print("\n--- Callbacks ---")

callbacks = [
    # Stop training when val_loss stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3,
        restore_best_weights=True, verbose=1,
    ),
    # Save the best model checkpoint
    tf.keras.callbacks.ModelCheckpoint(
        filepath="best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    # Reduce LR when a metric plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=2, min_lr=1e-6, verbose=1,
    ),
    # TensorBoard logging
    tf.keras.callbacks.TensorBoard(
        log_dir="./logs", histogram_freq=1,
    ),
    # Log metrics to CSV
    tf.keras.callbacks.CSVLogger("training_log.csv"),
]

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=0,
)

# Custom callback
class LRPrinter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        print(f"\nEpoch {epoch+1} — LR: {float(lr):.6f}")


# ============================================================
# 10. LEARNING RATE SCHEDULES
# ============================================================

print("\n--- LR Schedules ---")

# Cosine decay
cosine = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
)

# Exponential decay
exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=500,
    decay_rate=0.96,
    staircase=True,
)

# Polynomial decay
poly = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    end_learning_rate=1e-6,
    power=2.0,
)

# Warmup + cosine (custom schedule)
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps):
        super().__init__()
        self.peak_lr      = peak_lr
        self.warmup_steps = warmup_steps
        self.cosine       = tf.keras.optimizers.schedules.CosineDecay(
            peak_lr, total_steps - warmup_steps,
        )

    def __call__(self, step):
        warmup_lr = self.peak_lr * (step / self.warmup_steps)
        cosine_lr = self.cosine(step - self.warmup_steps)
        return tf.cond(step < self.warmup_steps,
                       lambda: warmup_lr, lambda: cosine_lr)

# Pass a schedule as the optimizer's learning_rate
optimizer = tf.keras.optimizers.Adam(learning_rate=cosine)


# ============================================================
# 11. CUSTOM TRAINING LOOP
# ============================================================
# For maximum control — replaces model.fit() entirely.

print("\n--- Custom Training Loop ---")

model2    = MyModel(num_classes=10)
optimizer = tf.keras.optimizers.Adam(1e-3)
loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy()

train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
train_acc_metric  = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")

# Build a tf.data pipeline
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(1000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

@tf.function                              # compile to a graph for speed
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model2(x, training=True)
        loss   = loss_fn(y, logits)
    grads = tape.gradient(loss, model2.trainable_variables)
    optimizer.apply_gradients(zip(grads, model2.trainable_variables))
    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(y, logits)

@tf.function
def val_step(x, y):
    logits = model2(x, training=False)
    return loss_fn(y, logits)

for epoch in range(1, 4):
    train_loss_metric.reset_state()
    train_acc_metric.reset_state()

    for x_batch, y_batch in train_ds:
        train_step(x_batch, y_batch)

    val_losses = [val_step(x, y) for x, y in val_ds]
    val_loss   = np.mean([l.numpy() for l in val_losses])

    print(f"Epoch {epoch} | "
          f"Loss: {train_loss_metric.result():.4f} | "
          f"Acc: {train_acc_metric.result():.4f} | "
          f"Val Loss: {val_loss:.4f}")


# ============================================================
# 12. tf.data — INPUT PIPELINES
# ============================================================

print("\n--- tf.data ---")

X_np = np.random.randn(500, 20).astype(np.float32)
y_np = np.random.randint(0, 10, 500)

# Build a dataset
ds = tf.data.Dataset.from_tensor_slices((X_np, y_np))

# Core transformations
ds = (
    ds
    .shuffle(buffer_size=500, seed=42)   # randomise order
    .batch(32)                           # group into batches
    .map(lambda x, y: (x * 2.0, y),     # apply transformation
         num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)          # overlap CPU prep with GPU training
    .repeat(3)                           # repeat dataset 3 times
)

# From files
# image_ds = tf.data.Dataset.list_files("images/*.jpg")
# text_ds  = tf.data.TextLineDataset("data.csv").skip(1)  # skip header
# tf.data.TFRecordDataset("data.tfrecord")                 # TFRecord format

# Cache to memory or disk
ds_cached = (
    tf.data.Dataset.from_tensor_slices((X_np, y_np))
    .cache()                             # cache in memory after first epoch
    .shuffle(500)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# 13. CONVOLUTIONAL NEURAL NETWORK (CNN)
# ============================================================

print("\n--- CNN ---")

cnn = tf.keras.Sequential([
    # Block 1
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu",
                           input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2),

    # Block 2
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2),

    # Classifier
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax"),
])

cnn.compile(optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
cnn.summary()

dummy_img = tf.random.normal([4, 28, 28, 1])
print("CNN output shape:", cnn(dummy_img).shape)


# ============================================================
# 14. TRANSFER LEARNING
# ============================================================

print("\n--- Transfer Learning ---")

# Load a pretrained base model (no top classification layers)
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,            # remove the ImageNet classifier
    weights="imagenet",           # pretrained weights
)

# Freeze the base
base.trainable = False

# Build a new classification head
inputs  = tf.keras.Input(shape=(224, 224, 3))
x       = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x       = base(x, training=False)          # frozen — always inference mode
x       = tf.keras.layers.GlobalAveragePooling2D()(x)
x       = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)  # 5 classes

transfer_model = tf.keras.Model(inputs, outputs)
transfer_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"])

print(f"Trainable params: {transfer_model.trainable_variables.__len__()}")

# Fine-tuning: unfreeze top layers of the base
base.trainable = True
fine_tune_at   = 100                       # unfreeze from layer 100 onward
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with a lower learning rate
transfer_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"])

# Available pretrained models:
#   tf.keras.applications.VGG16/VGG19
#   tf.keras.applications.ResNet50/ResNet101
#   tf.keras.applications.MobileNetV2/MobileNetV3
#   tf.keras.applications.EfficientNetB0..B7
#   tf.keras.applications.InceptionV3
#   tf.keras.applications.Xception


# ============================================================
# 15. SAVING & LOADING MODELS
# ============================================================

print("\n--- Save & Load ---")

# --- SavedModel format (recommended) ---
model.save("saved_model/my_model")                   # saves full model
loaded = tf.keras.models.load_model("saved_model/my_model")
print("SavedModel loaded:", loaded.predict(X_val[:2]).shape)

# --- Keras native format (.keras) ---
model.save("my_model.keras")
loaded2 = tf.keras.models.load_model("my_model.keras")

# --- Weights only ---
model.save_weights("weights.h5")
model.load_weights("weights.h5")

# --- Export for inference (TF SavedModel with signatures) ---
@tf.function(input_signature=[tf.TensorSpec([None, 20], tf.float32)])
def serve(x):
    return {"output": model(x, training=False)}

tf.saved_model.save(model, "serving_model",
                    signatures={"serving_default": serve})

# Clean up
import shutil
for path in ["saved_model", "my_model.keras", "weights.h5",
             "serving_model", "best_model.keras", "logs",
             "training_log.csv"]:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)


# ============================================================
# 16. TENSORBOARD
# ============================================================
# TensorBoard visualises training metrics, model graphs,
# weight histograms, embeddings, and more.

print("\n--- TensorBoard ---")

# 1. Add TensorBoard callback during training
log_dir = "logs/fit"
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,    # weight histograms every epoch
    write_graph=True,    # visualise model graph
    write_images=False,
    update_freq="epoch",
)
# model.fit(..., callbacks=[tb_callback])

# 2. Launch TensorBoard (from terminal):
#    tensorboard --logdir logs/fit
#    Then open http://localhost:6006 in a browser

# 3. Custom scalar logging with tf.summary
summary_writer = tf.summary.create_file_writer("logs/custom")
with summary_writer.as_default():
    for step in range(10):
        tf.summary.scalar("custom_metric", data=step * 0.1, step=step)
        tf.summary.histogram("random_weights",
                             data=tf.random.normal([100]), step=step)

# 4. In Jupyter / Colab
# %load_ext tensorboard
# %tensorboard --logdir logs/fit


# ============================================================
# QUICK REFERENCE CHEAT SHEET
# ============================================================
#
# TENSORS
#   tf.constant([...])                       Immutable tensor
#   tf.Variable([...])                       Mutable (trainable weights)
#   tf.zeros/ones/fill/eye(shape)            Constant tensors
#   tf.random.normal/uniform(shape)          Random tensors
#   tf.cast(t, tf.float32)                   Change dtype
#   t.numpy()                                To NumPy
#
# OPERATIONS
#   tf.add/subtract/multiply/matmul          Element-wise / matmul
#   tf.reduce_sum/mean/max/min(axis=)        Reductions
#   tf.reshape / tf.transpose                Shape manipulation
#   tf.squeeze / tf.expand_dims              Add/remove dims
#   tf.concat / tf.stack                     Combine tensors
#
# GRADIENTS
#   with tf.GradientTape() as tape:          Record operations
#   tape.gradient(loss, variables)           Compute gradients
#   optimizer.apply_gradients(zip(g, v))     Apply update
#
# MODELS
#   tf.keras.Sequential([layers])            Linear stack
#   tf.keras.Model(inputs, outputs)          Functional API
#   class MyModel(tf.keras.Model)            Subclassing API
#   model.summary()                          Print architecture
#
# COMMON LAYERS
#   Dense(units, activation)                 Fully connected
#   Conv2D(filters, kernel_size, padding)    Convolution
#   MaxPooling2D / GlobalAveragePooling2D    Pooling
#   LSTM / GRU / Bidirectional               Recurrent
#   BatchNormalization / LayerNormalization  Normalization
#   Dropout(rate)                            Regularization
#   Embedding(vocab_size, dim)               Embedding table
#   Flatten / Reshape / Concatenate          Utilities
#
# COMPILE & TRAIN
#   model.compile(optimizer, loss, metrics)  Configure training
#   model.fit(X, y, epochs, batch_size,      Train
#             validation_data, callbacks)
#   model.evaluate(X, y)                     Evaluate
#   model.predict(X)                         Inference
#
# LOSSES
#   SparseCategoricalCrossentropy            Multi-class (int labels)
#   CategoricalCrossentropy                  Multi-class (one-hot)
#   BinaryCrossentropy                       Binary classification
#   MeanSquaredError / MeanAbsoluteError     Regression
#   Huber                                    Robust regression
#
# CALLBACKS
#   EarlyStopping(monitor, patience)         Stop when plateau
#   ModelCheckpoint(filepath, monitor)       Save best model
#   ReduceLROnPlateau(monitor, factor)       Lower LR on plateau
#   TensorBoard(log_dir)                     Visualisation
#
# tf.data
#   Dataset.from_tensor_slices((X, y))       From NumPy/tensors
#   .shuffle(buffer).batch(n).prefetch(AUTO) Standard pipeline
#   .map(fn, num_parallel_calls=AUTO)        Transform
#   .cache()                                 Cache after first epoch
#
# TRANSFER LEARNING
#   tf.keras.applications.MobileNetV2(...)   Load pretrained
#   base.trainable = False                   Freeze
#   model.layers[i].trainable = True         Unfreeze selectively
#
# SAVE & LOAD
#   model.save("path/")                      SavedModel format
#   model.save("model.keras")                Keras native
#   model.save_weights("w.h5")               Weights only
#   tf.keras.models.load_model("path/")      Load
#
# TENSORBOARD
#   TensorBoard callback + tensorboard --logdir logs/
#   tf.summary.scalar/histogram(name, data, step)
