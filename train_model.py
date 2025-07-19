import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore

# Paths
DATA_PATH = os.path.join('data')
actions = np.array(sorted(os.listdir(DATA_PATH)))  # Sorted for consistency

print("⌛ Starting the optimized training process...")

# Load and normalize data
sequences, labels = [], []

for idx, action in enumerate(actions):
    action_path = os.path.join(DATA_PATH, action)
    for filename in os.listdir(action_path):
        if filename.endswith('.npy'):
            filepath = os.path.join(action_path, filename)
            seq = np.load(filepath)
            if seq.shape[0] == 30:
                # Normalize keypoints across each sample
                seq = (seq - np.mean(seq)) / (np.std(seq) + 1e-8)
                sequences.append(seq)
                labels.append(idx)

X = np.array(sequences)
y = tf.keras.utils.to_categorical(labels).astype(int)

# Save label map
label_map = {action: idx for idx, action in enumerate(actions)}
np.save('label_map.npy', label_map)

# ✅ Build model
model = Sequential()
model.add(LSTM(256, return_sequences=True, activation='tanh', input_shape=(30, X.shape[2])))
model.add(Dropout(0.5))
model.add(LSTM(256, return_sequences=True, activation='tanh'))
model.add(Dropout(0.5))
model.add(LSTM(128, activation='tanh'))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# ✅ Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
checkpoint = ModelCheckpoint('sign_model.h5', save_best_only=True, monitor='val_loss')

# ✅ Train
history = model.fit(
    X, y,
    epochs=300,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, lr_reduce, checkpoint],
    shuffle=True
)

print("✅ Training complete. Best model saved as 'sign_model.h5'")
