import numpy as np
from ct_model_getModelArchitecture import *
from dicomProcessing_helpingFunctions import *

from sklearn.model_selection import train_test_split


pos_dir = 'D:/CT-Dataset/positive'
neg_dir = 'D:/CT-Dataset/negative'

processed_type = str(input('Type: ', ))

data_pos = load_data(pos_dir,
                     # limit=10,
                     processed_type='numpy_data_processed' + processed_type)
data_neg = load_data(neg_dir,
                     # limit=10,
                     processed_type='numpy_data_processed' + processed_type)

# data_pos.dtype = 'float32'
# data_neg.dtype = 'float32'

# data_pos = data_pos.swapaxes(1, 3)
# data_neg = data_neg.swapaxes(1, 3)

target_positive = np.ones((len(data_pos),), dtype='int')
target_negative = np.zeros((len(data_neg),), dtype='int')

X = np.concatenate((data_pos, data_neg), axis=0)
y = np.concatenate((target_positive, target_negative), axis=0)

# x_train, x_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8, shuffle=True)
# x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, test_size=0.5, shuffle=True)


width, height, depth, n = X.T.shape
print(width, height, depth, n)
# model = get_model(width, height, depth)


initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

model.compile(loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"])

# Define callbacks.
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("3d_image_classification.h5", save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("Start training")

# Train the model, doing validation at the end of each epoch
epochs = 100
if False:
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint_cb, early_stopping_cb])


