import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K 
from tensorflow.keras import losses
from tensorflow.keras.datasets import mnist

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm

tf.compat.v1.disable_eager_execution()

print('keras:', keras.__version__)
print('tensorflow:', tf.__version__)
print('python:', sys.version)

batch_size = 100
original_dim = 28*28
intermediate_dim = 256
latent_dim = 2
nb_epoch = 5

### VAE

## Encoder

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0) # [100, 2]
    return z_mean + K.exp(z_log_var / 2) * epsilon

x = Input(shape=(original_dim), name="input")
h = Dense(intermediate_dim, activation='relu', name='encoding')(x)

z_mean = Dense(latent_dim, name='mean')(h)
z_log_var = Dense(latent_dim, name='log-variance')(h)

z = Lambda(sampling, output_shape=(latent_dim))([z_mean, z_log_var])

encoder = Model(x, [z_mean, z_log_var, z], name='encoder') # 起始張量x, 結束張量[z_mean, z_log_var, z]

## Decoder

input_decoder = Input(shape=(latent_dim), name='decoder_input')
decoder_h = Dense(intermediate_dim, activation='relu', name='decoder_h')(input_decoder)
x_decoded = Dense(original_dim, activation='sigmoid', name='flat_decoded')(decoder_h)

decoder = Model(input_decoder, x_decoded, name='decoder')

## Combine

output_combined = decoder(encoder(x)[2])
vae = Model(x, output_combined)
vae.summary()

def vae_loss(x, x_decoded_mean, z_log_var=z_log_var, z_mean=z_mean, original_dim=original_dim): # x:label, x_decoded_mean:predicted)
    
    xent_loss = original_dim * losses.binary_crossentropy(x, x_decoded_mean)
    kl_loss = -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) #相對entropy (KL divergence)
    return xent_loss + kl_loss

vae.compile(optimizer='rmsprop', loss=vae_loss)

## Train

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) #len(x_train)=60000; np.prod(x_train.shape[1:])=784
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train, x_train, shuffle=True, epochs=nb_epoch, batch_size=batch_size, validation_data=(x_test, x_test), verbose=1)

## display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)[0]
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c=y_test, cmap='viridis')
plt.colorbar()
plt.show()


## display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
## linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
## to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()

## VAE在生成圖片，給予input時可以巧妙地使用統計分布表達hidden space(類似EM-algorithm)，但缺點在於一般都是使用單峰Gaussion，實際上分布是未知的。

