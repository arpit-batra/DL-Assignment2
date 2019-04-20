import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, ReLU, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import tensorflow as tf
import keras_metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

sess = tf.Session()


def print_different_scores(cm,file):
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

	f= open(file+".txt","w")
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


# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

x_train=np.load('q1_data/x_train.npy')
y_train=np.load('q1_data/y_train.npy')
y_train_length = np.load('q1_data/y_train_length.npy')
y_train_width=np.load('q1_data/y_train_width.npy')
y_train_angle = np.load('q1_data/y_train_angle.npy')
y_train_color = np.load('q1_data/y_train_color.npy')
x_test=np.load('q1_data/x_test.npy')
y_test=np.load('q1_data/y_test.npy')
y_test_length = np.load('q1_data/y_test_length.npy')
y_test_width = np.load('q1_data/y_test_width.npy')
y_test_angle = np.load('q1_data/y_test_angle.npy')
y_test_color = np.load('q1_data/y_test_color.npy')


batch_size = 128
num_classes = 96
epochs = 5
train_example=int(x_train.shape[0]/1)
test_example=int(x_test.shape[0]/1)

# input image dimensions
image_rows, image_columns = 28, 28
image_size = 28*28*3


y_test1=y_test[0:test_example]

x_train_size = x_train.shape[0]
x_test_size = x_test.shape[0]


print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')
print(y_train.shape, 'train samples')
print(y_test.shape, 'test samples')

print(len(y_train_angle))
# for i in range(len(y_test_train_angle)):
# 	print(y_train_angle[i])

# convert class vectors to binary class matrices (hot encoding)
y_train_hot = keras.utils.to_categorical(y_train, num_classes)
y_train_length_hot = keras.utils.to_categorical(y_train_length, 2)
y_train_width_hot = keras.utils.to_categorical(y_train_width, 2)
y_train_angle_hot = keras.utils.to_categorical(y_train_angle, 12)
y_train_color_hot = keras.utils.to_categorical(y_train_color, 2)

y_test_hot = keras.utils.to_categorical(y_test, num_classes)
y_test_length_hot = keras.utils.to_categorical(y_test_length, 2)
y_test_width_hot = keras.utils.to_categorical(y_test_width, 2)
y_test_angle_hot = keras.utils.to_categorical(y_test_angle, 12)
y_test_color_hot = keras.utils.to_categorical(y_test_color, 2)

# print(y_train_length_hot[1901])
# print(y_train_width_hot[1901])
# print(y_train_angle_hot[1901])
# print(y_train_color_hot[1901])

# print(y_test_length_hot[1901])
# print(y_test_width_hot[1901])
# print(y_test_angle_hot[1901])
# print(y_test_color_hot[1901])


# here network structure begins

# for method-1
# x = tf.placeholder(tf.float32, [None,28,28,3])
# y_true = tf.placeholder(tf.float32, [None, num_classes])
# y_true_class = tf.placeholder(tf.int64, [None])



input = Input(shape=(28,28,3))

h1 = Conv2D(32, kernel_size=(7, 7), strides=(1, 1))(input)
h1 = ReLU()(h1)
# h1 = Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(input)
# h1 = ReLU()(h1)
# h1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1))(h1)
# h1 = ReLU()(h1)
h1 = BatchNormalization()(h1)
h1 = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(h1)



thread_1 = Flatten()(h1)
# thread_1 = Dense(1024)(thread_1)
# thread_1 = ReLU()(thread_1)
# thread_1 = Dropout(0.5)(thread_1)
# thread_1 = Dense(512)(thread_1)
# thread_1 = ReLU()(thread_1)
thread_1 = Dense(1024)(thread_1)
thread_1 = ReLU()(thread_1)
# thread_1 = Dense(32)(thread_1)
# thread_1 = ReLU()(thread_1)
# h1 = Flatten()(h1)
width_output = Dense(2, activation='sigmoid')(thread_1)

thread_2 = Flatten()(h1)
# thread_2 = Dense(1024)(thread_2)
# thread_2 = ReLU()(thread_2)
# thread_2 = Dropout(0.5)(thread_2)
thread_2 = Dense(1024)(thread_2)
thread_2	 = ReLU()(thread_2)
# h1 = Flatten()(h1)
color_output = Dense(2, activation='sigmoid')(thread_2)

thread_3 = Flatten()(h1)
# thread_3 = Dense(1024)(thread_3)
# thread_3 = ReLU()(thread_3)
# thread_3 = Dropout(0.5)(thread_3)
thread_3 = Dense(1024)(thread_3)
thread_3 = ReLU()(thread_3)
# h1 = Flatten()(h1)
length_output = Dense(2, activation='sigmoid')(thread_3)

thread_4 = Flatten()(h1)
# thread_4 = Dense(1024)(thread_4)
# thread_4 = ReLU()(thread_4)
# thread_4 = Dropout(0.5)(thread_4)
thread_4 = Dense(1024)(thread_4)
thread_4 = ReLU()(thread_4)
# h1 = Flatten()(h1)
angle_output = Dense(12, activation='softmax')(thread_4)

layer_width = 'dense_2'
layer_color = 'dense_4'
layer_length = 'dense_6'
layer_angle = 'dense_8'

losses = {
	layer_width: 'binary_crossentropy',
	layer_color: 'binary_crossentropy',
	layer_length: 'binary_crossentropy'
	,
	layer_angle: 'categorical_crossentropy'
}
metrics = {
	layer_width: 'accuracy',
	layer_color: 'accuracy',
	layer_length: 'accuracy'
	,
	layer_angle: 'accuracy'
}
lossWeights = {layer_width: 0.1, layer_color: 0.1, layer_length: 0.1, layer_angle: 0.6}

outputs=[width_output,color_output,length_output,angle_output]
X_test=x_test[0:test_example]
Y_train=[y_train_width_hot[0:train_example],y_train_color_hot[0:train_example],y_train_length_hot[0:train_example],y_train_angle_hot[0:train_example]]
Y_test=[y_test_width_hot[0:test_example],y_test_color_hot[0:test_example],y_test_length_hot[0:test_example],y_test_angle_hot[0:test_example]]

Y_test1={
	layer_width: y_test_width_hot[0:test_example],
	layer_color: y_test_color_hot[0:test_example],
	layer_length: y_test_length_hot[0:test_example]
	,
	layer_angle: y_test_angle_hot[0:test_example]
}

model = Model(inputs=input, outputs=outputs)
model.summary()
# [,keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()]
model.compile(loss=losses,loss_weights=lossWeights,optimizer=keras.optimizers.Adam(),metrics=metrics)
history = model.fit(x_train[0:train_example], y=Y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test,Y_test1),callbacks=[tbCallBack])
score = model.evaluate(X_test, y=Y_test1, verbose=0)
print(score)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
print(test_example)
pre_cls=model.predict(X_test)

# print((pre_cls[0]))
# print((pre_cls[1]))
# print((pre_cls[2]))

for i in range(len(pre_cls)):
	pre_cls[i]=np.argmax(pre_cls[i],axis=1)

# print((pre_cls[0]))
# print((pre_cls[1]))
# print((pre_cls[2]))

cm1 = confusion_matrix(y_test_width[0:test_example],pre_cls[0])
print (cm1)
print_different_scores(cm1,'width_data')
cm2 = confusion_matrix(y_test_color[0:test_example],pre_cls[1])
print (cm2)
print_different_scores(cm2,'color_data')
cm3 = confusion_matrix(y_test_length[0:test_example],pre_cls[2])
print (cm3)
print_different_scores(cm3,'length_data')
cm4 = confusion_matrix(y_test_angle[0:test_example],pre_cls[3])
# print (cm4)
print_different_scores(cm4,'angle_data')

# print_different_scores(cm1)


# print(history.history.keys())
# keys = list(history.history.keys())
# print(keys[0])
# for i in range(len(keys)):
# 	print(keys[i],'  --  ',history.history[keys[i]])


# saving figures

name='width head accuracy'
plt.plot(history.history['val_dense_2_acc'])
plt.xlim(0,epochs) 
plt.title(name)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('width_acc')
plt.clf()
# plt.show()

name='color head accuracy'
plt.plot(history.history['val_dense_4_acc'])
plt.xlim(0,epochs) 
plt.title(name)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('color_acc')
plt.clf()

name='length head accuracy'
plt.plot(history.history['val_dense_6_acc'])
plt.xlim(0,epochs) 
plt.title(name)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('length_acc')
plt.clf()

name='angle head accuracy'
plt.plot(history.history['val_dense_8_acc'])
plt.xlim(0,epochs) 
plt.title(name)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('angle_acc')
plt.clf()

plt.plot(history.history['val_dense_2_loss'])
plt.title('width head loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim(0,epochs) 
plt.savefig('width_loss')
plt.clf()

plt.plot(history.history['val_dense_4_loss'])
plt.title('color head loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim(0,epochs) 
plt.savefig('color_loss')
plt.clf()

plt.plot(history.history['val_dense_6_loss'])
plt.title('length head loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim(0,epochs) 
plt.savefig('length_loss')
plt.clf()

plt.plot(history.history['val_dense_8_loss'])
plt.title('angle head loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim(0,epochs) 
plt.savefig('angle_loss')
plt.clf()

