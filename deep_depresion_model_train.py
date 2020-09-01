'''
pip3 install pydot
pip3 install matplotlib
pip3 install graphviz
'''
import os
import numpy
import pandas
from tensorflow import keras

from tensorflow.keras.backend import expand_dims
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

condition_folder = list(os.walk("data/condition"))[0]
control_folder = list(os.walk("data/control"))[0]

condidtion_files = ["%s/%s"%(condition_folder[0],f) for f in condition_folder[2]]

control_files = ["%s/%s"%(control_folder[0],f) for f in control_folder[2]]

x = []
y = []

for f in condidtion_files:
	df = pandas.read_csv(f)
	x1 = numpy.array(df['activity'].tolist())
	x.append(x1)
	y.append(1)

for f in control_files:
	df = pandas.read_csv(f)
	x1 = numpy.array(df['activity'].tolist())
	x.append(x1)
	y.append(0)

x = numpy.array(x)
y = numpy.array(y)

seq_len_max = max([len(x1) for x1 in x])

x = pad_sequences(
	x, maxlen=seq_len_max, 
	dtype='int32', padding='pre', truncating='pre',
	value=0.0
	)

x = expand_dims(x, axis=-1)
y = to_categorical(y)

num_classes = y.shape[1]

numpy.save('x.npy', x)
numpy.save('y.npy', y)

x = numpy.load('x.npy')
y = numpy.load('y.npy')

'''
(65407, 1)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(x[0])
plt.show()
plt.close()
'''

def make_model(input_shape):
	input_layer = keras.layers.Input(input_shape)
	conv1 = keras.layers.Conv1D(filters=300, kernel_size=30, strides = 10, padding="valid")(input_layer)
	conv2 = keras.layers.Conv1D(filters=300, kernel_size=30, strides = 10, padding="valid")(conv1)
	conv2 = keras.layers.MaxPooling1D()(conv2)
	conv3 = keras.layers.Conv1D(filters=300, kernel_size=30, strides = 10, padding="valid")(conv2)
	gap = keras.layers.GlobalMaxPooling1D()(conv3)
	output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)
	return keras.models.Model(inputs=input_layer, outputs=output_layer)

model = make_model(input_shape=x.shape[1:])
model.compile(
	optimizer="adam",
	loss="categorical_crossentropy",
	metrics=["acc"],
	)

epochs = 10
batch_size = 60

acc = 0
while(acc <=0.99):
	history = model.fit(
		x,
		y,
		batch_size=batch_size,
		epochs=epochs,
		verbose=1,
	)
	acc = history.history['acc'][-1]

y_pred = model.predict(x)
numpy.argmax(y_pred, 1)

model.save("depression.h5")

model = keras.models.load_model("depression.h5")

test_loss, test_acc = model.evaluate(x, y
