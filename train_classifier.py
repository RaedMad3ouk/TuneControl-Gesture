import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Find the maximum sequence length
max_length = max(len(seq) for seq in data_dict['data'])

# Pad sequences to ensure uniform shape using keras pad_sequences
data_padded = pad_sequences(data_dict['data'], maxlen=max_length, padding='post', truncating='post', dtype='float32')
labels = np.asarray(data_dict['labels'])

# One-hot encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Save the label binarizer for inference
pickle.dump(lb, open('label_binarizer.p', 'wb'))

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# Define the neural network model with Dropout for regularization
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(max_length,)),
    keras.layers.Dropout(0.3),  # Adding Dropout to reduce overfitting
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),  # Adding Dropout to reduce overfitting
    keras.layers.Dense(len(lb.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up EarlyStopping and ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test),
          callbacks=[early_stopping, model_checkpoint])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the final model (best version already saved through ModelCheckpoint)
model.save('model.keras')

# Ensure proper input processing for inference
def preprocess_input(data_aux, max_length):
    data_aux = np.asarray(data_aux).reshape(1, -1)  # Ensure 2D shape
    data_aux = pad_sequences([data_aux], maxlen=max_length, padding='post', truncating='post', dtype='float32')
    return data_aux