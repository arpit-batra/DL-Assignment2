import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import tensorflow as tf
import keras_metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


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
	print ('recall = ',recall)
	print ('precision = ',precision)
	f_score=[]
	for i in range(len(recall)):
		val = 2.0 * recall[i] * precision[i]
		val /= (precision[i]+recall[i])
		f_score.append(val)

	print ('f_score = ',f_score)

batch_size = 128
num_classes = 10
epochs = 5
train_example=60000
test_example=10000

# input image dimensions
# img_rows, img_cols = 28, 28
image_rows, image_columns = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
y_test1=y_test[0:test_example]
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# print(y_train.shape[0], 'train samples')
# print(y_test.shape[0], 'test samples')


# convert class vectors to binary class matrices (hot encoding)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# print(y_train[0], 'train samples')
# print(y_test[0], 'test samples')

# Create a model with the above specified network architecture. Use the Adam optimizer
# with categorical crossentropy loss. Once the model is trained test it using the test data.

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))




# [,keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()]
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
# with tf.Session( config = tf.ConfigProto( log_device_placement = True ) ):
history = model.fit(x_train[0:train_example], y_train[0:train_example],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
          # validation_data=(x_test, y_test))

score = model.evaluate(x_test[0:test_example], y_test[0:test_example], verbose=0)
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

pre_cls=model.predict_classes(x_test[0:test_example])

cm1 = confusion_matrix(y_test1,pre_cls)
print('Confusion Matrix : \n', cm1)
print_different_scores(cm1)

# history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
ax = plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# print(ax)
plt.xlim(0,epochs) 
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim(0,epochs) 

# plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Test loss: 0.041282033716811564
# Test accuracy: 0.9872
# Confusion Matrix : 
#  [[ 972    1    1    0    1    0    2    1    1    1]
#  [   0 1132    1    1    0    0    1    0    0    0]
#  [   2    3 1018    0    3    0    0    6    0    0]
#  [   0    0    1 1004    0    2    0    1    1    1]
#  [   0    0    0    0  976    0    1    0    0    5]
#  [   2    0    0    6    0  880    3    0    0    1]
#  [   3    4    0    0    1    2  946    0    2    0]
#  [   0    2    8    1    0    0    0 1013    1    3]
#  [   9    1    3    3    3    1    1    1  946    6]
#  [   2    6    0    2    6    3    0    5    0  985]]
# accuracy =  0.9872
# recall =  [0.9918367346938776, 0.9973568281938326, 0.9864341085271318, 0.994059405940594, 0.9938900203665988, 0.9865470852017937, 0.9874739039665971, 0.9854085603112841, 0.971252566735113, 0.9762140733399405]
# precision =  [0.9818181818181818, 0.9852045256744996, 0.9864341085271318, 0.9872173058013766, 0.9858585858585859, 0.990990990990991, 0.9916142557651991, 0.9863680623174295, 0.9947423764458465, 0.9830339321357285]
# f_score =  [0.9868020304568528, 0.9912434325744308, 0.9864341085271318, 0.9906265416872225, 0.9898580121703855, 0.9887640449438202, 0.9895397489539749, 0.9858880778588808, 0.982857142857143, 0.9796121332670313]





