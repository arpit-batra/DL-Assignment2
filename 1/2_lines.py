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
	# print ('recall = ',recall)
	# print ('precision = ',precision)
	f_score=[]
	for i in range(len(recall)):
		val = 2.0 * recall[i] * precision[i]
		val /= (precision[i]+recall[i])
		f_score.append(val)

	# print ('f_score = ',f_score)
	f= open("2_lines_data.txt","w")
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
			f.write(str(float("{0:.3f}".format(recall[j])))+' & ')
		else:
			f.write(str(float("{0:.3f}".format(recall[j])))+" \\"+"\\")
	f.write('\n')

	f.write('precision\n')
	for j in range(len(precision)):
		if(j<len(precision)-1):
			f.write(str(float("{0:.3f}".format(precision[j])))+' & ')
		else:
			f.write(str(float("{0:.3f}".format(precision[j])))+" \\"+"\\")
	f.write('\n')
	
	f.write('f_score\n')
	for j in range(len(f_score)):
		if(j<len(f_score)-1):
			f.write(str(float("{0:.3f}".format(f_score[j])))+' & ')
		else:
			f.write(str(float("{0:.3f}".format(f_score[j])))+" \\"+"\\")
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
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
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
plt.plot(history.history['acc'])
plt.xlim(0,epochs) 
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('2_lines_acc')
plt.clf()
# plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim(0,epochs) 
plt.savefig('2_lines_loss')
# plt.show()

# [0.9569928871570543, 0.8591666666666666]
# Test loss: 0.9569928871570543
# Test accuracy: 0.8591666666666666
# [[ 40   0   0 ...   0   0   0]
#  [  0  82   0 ...   0   0   0]
#  [  0   0  99 ...   1   0   0]
#  ...
#  [  0   0   6 ...  94   0   0]
#  [  0   0   0 ...   0 100   0]
#  [  0   0   0 ...   0   0 100]]
# 1_lines.py:28: RuntimeWarning: invalid value encountered in double_scalars
#   precision_val = (1.0*cm[i][i]/cm[:,i].sum());
# recall =  [0.4, 0.82, 0.99, 1.0, 1.0, 1.0, 1.0, 0.58, 0.99, 0.96, 0.53, 1.0, 1.0, 0.89, 0.97, 0.52, 1.0, 0.9, 1.0, 0.87, 0.94, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.72, 0.94, 0.66, 1.0, 1.0, 0.76, 0.91, 1.0, 1.0, 1.0, 0.83, 1.0, 0.56, 0.5, 1.0, 0.36, 1.0, 0.61, 1.0, 0.97, 0.96, 0.91, 1.0, 1.0, 1.0, 0.97, 0.82, 1.0, 0.92, 0.99, 0.73, 1.0, 1.0, 0.72, 1.0, 1.0, 0.93, 0.0, 1.0, 1.0, 0.76, 0.99, 0.91, 1.0, 0.65, 1.0, 1.0, 0.97, 1.0, 0.94, 0.85, 0.63, 0.64, 0.0, 1.0, 0.6, 0.93, 0.74, 0.96, 0.95, 0.93, 0.96, 1.0, 1.0, 0.94, 1.0, 1.0]
# precision =  [1.0, 0.6456692913385826, 0.8114754098360656, 1.0, 0.7575757575757576, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9814814814814815, 1.0, 1.0, 0.967391304347826, 1.0, 0.8387096774193549, 0.6097560975609756, 0.782608695652174, 1.0, 0.9886363636363636, 1.0, 1.0, nan, 1.0, 1.0, 0.9009009009009009, nan, 1.0, 1.0, 0.7741935483870968, 0.7642276422764228, 0.7857142857142857, 0.8928571428571429, 1.0, 0.8539325842696629, 1.0, 1.0, 0.7936507936507936, 0.45045045045045046, 0.8556701030927835, 0.8, 0.7272727272727273, 0.5263157894736842, 1.0, 0.9, 1.0, 0.648936170212766, 1.0, 0.97, 0.6620689655172414, 0.978494623655914, 1.0, 1.0, 1.0, 0.60625, 0.9010989010989011, 1.0, 0.9484536082474226, 1.0, 0.8295454545454546, 0.7407407407407407, 0.5025125628140703, 0.6206896551724138, 1.0, 1.0, 0.7099236641221374, nan, 1.0, 1.0, 0.6440677966101694, 0.8608695652173913, 0.7459016393442623, 0.7874015748031497, 0.9027777777777778, 1.0, 1.0, 1.0, 1.0, 0.6573426573426573, 0.5214723926380368, 1.0, 0.9552238805970149, nan, 1.0, 0.5454545454545454, 0.8532110091743119, 1.0, 0.9696969696969697, 0.7851239669421488, 0.9207920792079208, 0.8, 1.0, 0.9345794392523364, 0.9494949494949495, 1.0, 1.0]
# f_score =  [0.5714285714285715, 0.7224669603524229, 0.8918918918918919, 1.0, 0.8620689655172413, 1.0, 1.0, 0.7341772151898733, 0.9949748743718593, 0.9795918367346939, 0.6883116883116882, 1.0, 1.0, 0.9270833333333334, 0.9847715736040609, 0.6419753086419753, 0.7575757575757575, 0.8372093023255814, 1.0, 0.925531914893617, 0.9690721649484536, 1.0, nan, 1.0, 1.0, 0.947867298578199, nan, 1.0, 1.0, 0.7461139896373057, 0.8430493273542602, 0.717391304347826, 0.9433962264150945, 1.0, 0.8042328042328041, 0.9528795811518325, 1.0, 0.8849557522123894, 0.6211180124223602, 0.8426395939086294, 0.888888888888889, 0.632768361581921, 0.5128205128205129, 1.0, 0.5142857142857143, 1.0, 0.6288659793814433, 1.0, 0.97, 0.7836734693877551, 0.9430051813471503, 1.0, 1.0, 1.0, 0.7461538461538461, 0.8586387434554974, 1.0, 0.9340101522842639, 0.9949748743718593, 0.7765957446808511, 0.851063829787234, 0.6688963210702341, 0.6666666666666666, 1.0, 1.0, 0.8051948051948052, nan, 1.0, 1.0, 0.6972477064220184, 0.9209302325581394, 0.8198198198198198, 0.881057268722467, 0.7558139534883721, 1.0, 1.0, 0.9847715736040609, 1.0, 0.7736625514403291, 0.6463878326996197, 0.7730061349693252, 0.7664670658682635, nan, 1.0, 0.5714285714285713, 0.8899521531100477, 0.8505747126436781, 0.964824120603015, 0.8597285067873304, 0.9253731343283582, 0.8727272727272728, 1.0, 0.966183574879227, 0.9447236180904524, 1.0, 1.0]