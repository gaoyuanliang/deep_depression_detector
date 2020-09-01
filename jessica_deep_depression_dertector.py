import os
import numpy
import pandas
from tensorflow import keras

from tensorflow.keras.backend import expand_dims
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = keras.models.load_model("depression.h5")

def deep_depression_detector(activity_data_csv_file):
	df = pandas.read_csv(activity_data_csv_file)
	x = numpy.array([df['activity'].tolist()])
	x = pad_sequences(
		x, maxlen=65407, 
		dtype='int32', padding='pre', truncating='pre',
		value=0.0
		)
	x = expand_dims(x, axis=-1)
	y_pred = model.predict(x)[0]
	y_score = numpy.max(y_pred)
	y_label = numpy.argmax(y_pred)
	if y_label == 1:
		return  {'prediction':'depressed', 'confidence': y_score}
	else:
		return  {'prediction':'nondepressed', 'confidence': y_score}
