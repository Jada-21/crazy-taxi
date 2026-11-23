import tensorflow as tf
from tensorflow . keras import layers , models
model = models.Sequential ([

#Jada Angueira 

layers.Input(shape=(227, 227, 3)),

#conv1
#96 filters, 11*11 kernel, stride = 4, relu activation, input shape of 227*227*3
layers.Conv2D (96, kernel_size=((11,11)), strides =4, activation = 'relu', input_shape = (227,227,3)),
layers.MaxPooling2D(pool_size=(3, 3), strides=2),

#conv2
#256 filters, 5*5 kernel, stride = same
layers.Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu'),
layers.MaxPooling2D(pool_size=(3, 3), strides=2),

#conv3
#384 filters, 3*3 kernel, stride = same
layers.Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),

#conv4
#384 filters, 3*3 kernel, stride = same
layers.Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),

#conv5
#256 filters, 3*3 kernel, stride = same
layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
layers.MaxPooling2D(pool_size=(3, 3), strides=2),

layers.Flatten(),

#connected layer
layers.Dense (4096 , activation = 'relu'), #FC6 
layers.Dense (4096 , activation = 'relu') ,#FC7
layers.Dense (10 , activation = 'softmax') ,#FC8

])
model.compile (optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' ,
metrics =['accuracy']

)
model.summary ()