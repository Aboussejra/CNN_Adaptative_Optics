# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Boussejra Amir
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to stop error messages

import pandas as pd # pandas 1.2.3
import numpy as np #1.19.2
data = pd.read_csv("calcAiry_1000frames_32px.csv", header=None)

taille_data = len(data)
percent_train = 0.8
taille_image = 32 # images carrées
nb_coefs = 20 # must be lower than the number of coefs given in dataset (here 56)
resize_factor = 10**-7 # tool used to resize the zernike coefs from 0 to 100, to better see the small ones

train =  data.iloc[0:int(taille_data*percent_train),:]
test = data.iloc[int(taille_data*percent_train):taille_data,:]

images_train = train.iloc[:,0:taille_image**2]
coefs_train = train.iloc[:,taille_image**2:taille_image**2 + nb_coefs]


images_test = test.iloc[:,0:taille_image**2]
coefs_test = test.iloc[:,taille_image**2:taille_image**2 + nb_coefs]

coefs_train_numpy = np.asanyarray(coefs_train)/(resize_factor) # coefs as numpy, bcs we convert images as tab (in data frames) to numpy matrices. Better to keep same data type
coefs_test_numpy = np.asanyarray(coefs_test)/(resize_factor)

def convert_image(tab):
    # convert a tab of length 256 as a 16*16 matrix
    image = np.zeros((taille_image,taille_image))
    # -1 is necessarly not in an image so why not
    n = len(tab) # here 256
    size = int(np.sqrt(n)) # numpy give 16.0 and that block len() object
    for i in range(0,size):
        for j in range(0,size):    
            image[i,j] = tab[j + i*size]
    return image
        
def convert_all_images(dataframe_images):
    n = len(dataframe_images)
    len_image = taille_image
    liste_images = np.zeros((n,len_image,len_image))
    for i in range(0,n):
        image_i = dataframe_images.iloc[i,:]
        liste_images[i] = convert_image(image_i)
    return liste_images
        
# converting images 
images_train_matrices = convert_all_images(images_train)
images_test_matrices = convert_all_images(images_test)

import tensorflow as tf # 2.2.0 print(tf.__version__)
tf.random.set_seed(1234)



# Training Parameters, to change here, i tried 128 batch size and 0.3 dropout, 100 epochs too. Here are the parameters used in OSA publishing
learning_rate = 10**-4
batch_size = 128
epochs = 20
dropout_ratio = 0.3

# classical architechure, CNN, then dropout to prevent overfitting and fully connected layer ending in desired output size 
# best result with other model with dropout before last layer.
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 16x16 with 1 bytes colour, neural networks perform better with input size between 0 and 1, so e rescale
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(taille_image, taille_image, 1)),
    #conv layers use the first number (16 then 32 then 64) of filters of size 2*2
    # filters helps detectings patterns, edges ect..
    # output of conv layers is bigger than input (result of filter as a bonus)
    tf.keras.layers.Conv2D(taille_image, kernel_size = (2,2), activation='relu'),
    # we augment the number of filters because In every layer filters are there to capture patterns. For example in the first layer filters 
    # capture patterns like edges, corners, dots etc.
    # In the subsequent layers we combine those patterns to make bigger patterns.
    # Like combine edges to make squares, circle etc.
    # Now as we move forward in the layers the patterns gets more complex, hence larger combinations of patterns to capture. 
    # That's why we increase filter size in the subsequent layers to capture as many combinations as possible.
    # still it is hard to know what pattern you detect in each layer
    # The higher the number of filters, the higher the number of abstractions the network is able to extract from image data
    tf.keras.layers.Conv2D(32, kernel_size = (2,2), activation='relu'),
    tf.keras.layers.Conv2D(64, kernel_size = (2,2), activation='relu'),
    # max pooling scrolls inside its input and picks max value out of a matrix of size 2*2
    # useful to keep only the fundamental distorsions (average pooling possible)
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    # dropouts disable some neurons at each steps, it helps preventing overfitting
    # To explain it simply, Neurons cannot rely on other units to correct their mistakes because of dropout so they caliber better globally.
    tf.keras.layers.Dropout(dropout_ratio),
    # flatten transform the input that is as an image to input it in a dense network
    tf.keras.layers.Flatten(), 
    # usual dense layers to interpolate what patterns has been found in images to numbers, dropout
    tf.keras.layers.Dense(256, activation='relu'), 
    tf.keras.layers.Dropout(dropout_ratio),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(dropout_ratio),
    tf.keras.layers.Dense(nb_coefs, activation='tanh')
    # relu is good because simple for computer, good for backpropagating in deep networks like this one
    # then it is capable of outputing true 0 (while sigmoid and tanh approximates it)
    # we want to guess true 0 when coefs are null, and output can be any number which is good
    
])

model.summary()
# learning rate would be useful if we want to define a custom optimizer as : 
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# here, automatic adam optimizer by tf looks better than the one in the paper in this experiment

# adam gradient descent, loss calculated with MSE, metrics to train the same, maybe useful to change
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

history = model.fit(images_train_matrices, coefs_train_numpy, batch_size=batch_size, epochs=epochs, validation_data=(images_test_matrices, coefs_test_numpy))

# saving model
# model.save('model_V1')

# memory of losses
#print(history.history['val_loss'])
test_pred = model.predict(images_test_matrices)

# a graphical test
pred_0 = test_pred[0]
test_0 = coefs_test_numpy[0]
liste_coefs = [i for i in range(0,nb_coefs)]

import matplotlib.pyplot as plt

plt.plot([i for i in range (0,epochs)],history.history['val_loss'])
plt.xlabel('epochs')
plt.title('MSE on validation set (real MSE is this one divided by 10⁻16)')
plt.show()

plt.scatter(liste_coefs,pred_0,label='prediction')
plt.scatter(liste_coefs,test_0,label='test')
plt.xlabel('coef number')
plt.ylabel('value')
plt.legend(loc='upper left')
plt.title(label = 'Exemple prediction')
plt.show()



