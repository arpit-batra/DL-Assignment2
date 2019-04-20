import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,ReLU
from keras.layers import Conv2D,MaxPooling2D
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
		# print ('row_sum= ',row_sum)
		# print ('col_sum= ',col_sum)

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

# batch_size =

num_classes=10
epochs=1
batch_size=128
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train=x_train.reshape((60000,28,28,1))
x_test=x_test.reshape((10000,28,28,1))
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

print(x_train.shape)
model = Sequential()
# 1. 7x7 Convolutional Layer with 32 filters and stride of 1.
model.add(Conv2D(32, kernel_size=(7, 7), strides=(1, 1), input_shape=(28,28,1)))
# 2. ReLU Activation Layer.
model.add(ReLU())
# 3. Batch Normalization Layer
model.add(BatchNormalization())
# 4. 2x2 Max Pooling layer with a stride of 2
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
# 5. fully connected layer with 1024 output units.
model.add(Dense(1024))
# 6. ReLU Activation Layer.
model.add(ReLU())
# final layer with output neurons same as no. of classes
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# [,keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()]
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=1)


#

# Test loss: 0.041458422699323545
# Test accuracy: 0.9896
# Confusion Matrix : 
#  [[ 976    0    1    0    0    1    1    1    0    0]
#  [   0 1133    2    0    0    0    0    0    0    0]
#  [   1    4 1023    0    0    0    0    4    0    0]
#  [   0    0    3  992    0   11    0    1    0    3]
#  [   0    1    1    0  976    0    3    0    0    1]
#  [   2    0    0    0    0  889    1    0    0    0]
#  [   8    3    0    0    7    4  936    0    0    0]
#  [   0    2    4    3    0    0    0 1018    0    1]
#  [   6    2   16    1    7   10    3    3  916   10]
#  [   2    5    2    1   12   10    0    3    0  974]]
# accuracy =  0.9833
# recall =  [0.9959183673469387, 0.9982378854625551, 0.9912790697674418, 0.9821782178217822, 0.9938900203665988, 0.9966367713004485, 0.9770354906054279, 0.9902723735408561, 0.9404517453798767, 0.9653121902874133]
# precision =  [0.9809045226130654, 0.9852173913043478, 0.9724334600760456, 0.9949849548645938, 0.9740518962075848, 0.961081081081081, 0.9915254237288136, 0.9883495145631068, 1.0, 0.9848331648129424]
# f_score =  [0.9883544303797468, 0.9916849015317286, 0.9817658349328214, 0.9885401096163429, 0.9838709677419355, 0.9785360484314805, 0.9842271293375394, 0.9893100097181731, 0.9693121693121692, 0.9749749749749749]

