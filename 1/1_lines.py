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

	f= open("1_lines_data.txt","w")
	f.write('confusion_matrix\n')
	for i in range(len(cm)):
		for j in range(len(cm[0])):
			if(j<len(cm[0])-1):
				f.write(str(cm[i][j])+' & ')
			else:
				f.write(str(cm[i][j])+" \\"+"\\")
		f.write('\n')
		
	f.write('recall\n')
	for j in range(len(recall)):
		if(j<len(recall)-1):
			f.write(str(recall[j])+' & ')
		else:
			f.write(str(recall[j])+" \\"+"\\")
	f.write('\n')

	f.write('precision\n')
	for j in range(len(precision)):
		if(j<len(precision)-1):
			f.write(str(precision[j])+' & ')
		else:
			f.write(str(precision[j])+" \\"+"\\")
	f.write('\n')
	
	f.write('f_score\n')
	for j in range(len(f_score)):
		if(j<len(f_score)-1):
			f.write(str(f_score[j])+' & ')
		else:
			f.write(str(f_score[j])+" \\"+"\\")
	f.write('\n')

	# end-function

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train=np.load('q1_data/x_train.npy')
y_train=np.load('q1_data/y_train.npy')
x_test=np.load('q1_data/x_test.npy')
y_test=np.load('q1_data/y_test.npy')


batch_size = 128
num_classes = 96
epochs = 5
train_example=int(x_train.shape[0]/1)
test_example=int(x_test.shape[0]/1)

# input image dimensions
# img_rows, img_cols = 28, 28
image_rows, image_columns = 28, 28


y_test1=y_test[0:test_example]
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_size = x_train.shape[0]
x_test_size = x_test.shape[0]

# x_train = x_train.reshape(x_train_size,28*28*3)
# x_test = x_test.reshape(x_test_size,28*28*3)

print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')

print(y_train.shape, 'train samples')
print(y_test.shape, 'test samples')


# convert class vectors to binary class matrices (hot encoding)
y_train_hot = keras.utils.to_categorical(y_train, num_classes)
y_test_hot = keras.utils.to_categorical(y_test, num_classes)

print('x_train shape:', x_train_size)
print('y_train shape:', x_test_size)
# print('x_test shape:', x_test.shape)
# print('y_test_hot shape:', y_test.shape)

print(y_train[0], 'train samples')
print(y_test[0], 'test samples')

# Create a model with the above specified network architecture. Use the Adam optimizer
# with categorical crossentropy loss. Once the model is trained test it using the test data.

model = Sequential()
# 1. 7x7 Convolutional Layer with 32 filters and stride of 1.
model.add(Conv2D(32, kernel_size=(7, 7), strides=(1, 1), input_shape=(28,28,3)))
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



# [,keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()]
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

history = model.fit(x_train[0:train_example], y_train_hot[0:train_example],
		  batch_size=batch_size,
		  epochs=epochs,
		  verbose=1)
		  # validation_data=(x_test, y_test_hot))

score = model.evaluate(x_test[0:test_example], y_test_hot[0:test_example], verbose=0)
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

pre_cls=model.predict_classes(x_test[0:test_example])

print(test_example)

# print(pre_cls)

cm1 = confusion_matrix(y_test1,pre_cls)
print(cm1)

print_different_scores(cm1)


print(history.history.keys())

plt.xlim(0,epochs) 
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('1_lines_acc')
plt.clf()
plt.close()
# plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim(0,epochs) 
plt.savefig('1_lines_loss')
# plt.show()


# Test loss: 0.02147064186883052
# Test accuracy: 0.9935416666666667
# 19200
# [[197   0   0 ...   0   0   0]
#  [  0 200   0 ...   0   0   0]
#  [  0   0 200 ...   0   0   0]
#  ...
#  [  0   0   0 ... 200   0   0]
#  [  0   0   0 ...   0 200   0]
#  [  0   0   0 ...   0   0 200]]
# accuracy =  0.9935416666666667
# recall =  [0.985, 1.0, 1.0, 1.0, 0.985, 1.0, 1.0, 1.0, 1.0, 1.0, 0.985, 1.0, 1.0, 0.935, 0.98, 1.0, 1.0, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.995, 1.0, 1.0, 1.0, 1.0, 0.96, 1.0, 0.96, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.945, 1.0, 1.0, 1.0, 1.0, 0.99, 1.0, 0.995, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.995, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.925, 0.98, 0.965, 0.985, 0.975, 0.97, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.905, 1.0, 1.0, 1.0, 0.985, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# precision =  [0.9609756097560975, 0.9900990099009901, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9751243781094527, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9320388349514563, 0.966183574879227, 1.0, 1.0, 1.0, 0.9803921568627451, 1.0, 1.0, 0.9569377990430622, 1.0, 1.0, 1.0, 0.9478672985781991, 1.0, 0.9852216748768473, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9950248756218906, 1.0, 1.0, 0.9707317073170731, 1.0, 1.0, 1.0, 1.0, 0.975609756097561, 1.0, 1.0, 1.0, 1.0, 0.9852216748768473, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9174311926605505, 1.0, 1.0, 1.0, 0.9345794392523364, 1.0, 0.9389671361502347, 0.9949494949494949, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# f_score =  [0.9728395061728395, 0.9950248756218906, 1.0, 1.0, 0.9924433249370278, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9924433249370278, 1.0, 1.0, 0.9664082687338501, 0.98989898989899, 1.0, 1.0, 0.9775561097256857, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9974937343358395, 1.0, 1.0, 1.0, 1.0, 0.9458128078817734, 0.9828009828009828, 0.9795918367346939, 1.0, 1.0, 0.99009900990099, 1.0, 1.0, 0.9779951100244498, 1.0, 1.0, 0.9717223650385605, 0.9732360097323601, 1.0, 0.9925558312655086, 1.0, 0.9949748743718593, 1.0, 0.9974937343358395, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9975062344139651, 1.0, 1.0, 0.982716049382716, 1.0, 1.0, 1.0, 1.0, 0.9876543209876543, 1.0, 1.0, 1.0, 1.0, 0.9925558312655086, 1.0, 0.961038961038961, 0.98989898989899, 0.9821882951653944, 0.9924433249370278, 0.9873417721518987, 0.9847715736040609, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9569377990430622, 1.0, 1.0, 0.9501312335958005, 0.966183574879227, 1.0, 0.9685230024213075, 0.9899497487437187, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]