import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import tensorflow as tf


def print_different_scores(cm):
	# print(cm)
	recall=[]
	precision=[]
	recall_val = 0
	accuracy=0
	total_sum=0
	for i in range(len(cm)):
		num = cm[i][i]
		accuracy+=num
		row_sum=cm[i].sum()
		col_sum=cm[:,i].sum()
		total_sum+=row_sum
		# print ('row_sum= ', row_sum)
		# print ('col_sum= ', col_sum)

		recall_val = (1.0*num/row_sum);
		recall.append(recall_val);
		precision_val = (1.0*cm[i][i]/cm[:,i].sum());
		precision.append(precision_val);

	accuracy = (1.0*accuracy/total_sum)
	print('accuracy = ',accuracy)
	# print ('recall = ',recall)
	# print ('precision = ',precision)
	f_score=[]
	for i in range(len(recall)):
		val = 2.0 * recall[i] * precision[i]
		val /= (precision[i]+recall[i])
		f_score.append(val)

	# print ('f_score = ',f_score)

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
# img_rows, img_cols = 28, 28
image_rows, image_columns = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# print(y_train[0], 'train samples')
# print(y_test[0], 'test samples')


# convert class vectors to binary class matrices (hot encoding)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# print(y_train[0], 'train samples')
# print(y_test[0], 'test samples')


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])