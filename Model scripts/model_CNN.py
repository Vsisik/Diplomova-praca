import numpy as np

from model_getModelArchitecture import *
from dicomProcessing_helpingFunctions import *

pos_dir = ''
neg_dir = ''

data_pos = load_data(pos_dir)
data_neg = load_data(neg_dir)

target_positive = np.ones((len(data_pos),), dtype='int')
target_negative = np.zeros((len(data_neg),), dtype='int')

X = np.concatenate((data_pos, data_neg), axis=0)
y = np.concatenate((target_positive, target_negative), axis=0)

x_train, x_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8, shuffle=True)
x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, test_size=0.5, shuffle=True)


width, height, depth, n = X.T.shape
model = get_model(width, height, depth)
