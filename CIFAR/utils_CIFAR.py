import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.models import Model

import keras
# from keras.datasets import mnist
from keras.datasets import cifar10

from tqdm import tqdm_notebook as tqdm
from keras.layers.advanced_activations import LeakyReLU

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from numpy import expand_dims
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras import backend
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.constraints import Constraint
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import _Merge
from skimage.transform import resize

# from keras.optimizers import Optimizer
# from tensorflow.keras import backend 
from keras.optimizers import Adam, SGD

from tqdm.notebook import tqdm
from functools import partial
from matplotlib.image import imread
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm

# The Players class defines the two-player game and all necessary update functions
# x - min player
# y - max player
# f - a function of x, y to create new entity z
# u_x - function to take gradient update step of min player x
# u_y - function to take gradient update step of max player y
# c_x - function to change value of x to x_new
# c_y - function to change value of y to y_new
# p_x - function to perturb value of x along a random normal direction

class Players:
    def __init__(self, x, y, f, u_x, u_y, c_x, c_y, p_x):
        self.x = x
        self.y = y
        if f == None:
            self.z = None
        else:
            self.z = f(x, y)
        self.u_x = u_x
        self.u_y = u_y
        self.c_x = c_x
        self.c_y = c_y
        self.p_x = p_x
    
    def value(self, f):
        return f(self.x, self.y)

    def get_x(self):
        return self.x
                
    def get_y(self):
        return self.y       

    def update_x(self):
        self.x = self.u_x(self.x, self.y, self.z)
        return self.x
        
    def update_y(self):
        self.y = self.u_y(self.x, self.y, self.z)
        return self.y
    
    def change_x(self, x_new):
        self.x = self.c_x(self.x, x_new)
                
    def change_y(self, y_new):
        self.y = self.c_y(self.y, y_new)

    def perturb_x(self):
        self.x = self.p_x(self.x)
        return self.x
    

# img = imread('image_digit_1.png')
# img = img.reshape(400)

batch_size = 64

def getGDopt(lr = 0.01):
    return SGD(lr)

# Load CIFAR data   
def load_data(filter=True):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    
#     x_train = x_train.reshape(len(x_train), 784)
            
    return (x_train, y_train, x_test, y_test)

# (X_train, y_train, X_test, y_test) = load_data()
# X_train = np.array([img]*128)


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = backend.random_uniform((batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    gradients = backend.gradients(backend.sum(y_pred), averaged_samples)
    gradient_l2_norm = backend.sqrt(backend.sum(backend.square(gradients)))
    gradient_penalty = gradient_penalty_weight * backend.square(1 - gradient_l2_norm)
    return gradient_penalty
    
# Create generator network with preferred optimization function
def create_generator(OUTPUT_SIZE, opt=getGDopt(), INPUT_SIZE=100, loss='binary_crossentropy'):
    generator = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    generator.add(Dense(n_nodes, input_dim=INPUT_SIZE))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Reshape((4, 4, 256)))
    # upsample to 8x8
    generator.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))
    # upsample to 16x16
    generator.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    generator.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))
    # output layer
    generator.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
        
    if loss == 'binary_crossentropy':
        generator.compile(loss='binary_crossentropy', optimizer=opt)
    else:
        generator.compile(loss=wasserstein_loss, optimizer=opt, metrics=['accuracy'])        
    return generator

# Create discriminator network with preferred optimization function
def create_discriminator(INPUT_SHAPE=(32,32,3), opt=getGDopt(), loss='binary_crossentropy'):
    discriminator = Sequential()
    # normal
    discriminator.add(Conv2D(64, (3,3), padding='same', input_shape=INPUT_SHAPE))
    discriminator.add(LeakyReLU(alpha=0.2))
    # downsample
    discriminator.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    # downsample
    discriminator.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    # downsample
    discriminator.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    # classifier
    discriminator.add(Flatten())
    discriminator.add(Dropout(0.4))
    if loss=='binary_crossentropy':
        discriminator.add(Dense(units=1, activation='sigmoid'))
    else:
        discriminator.add(Dense(units=1))        
    
    discriminator.trainable = True
    if loss == 'binary_crossentropy':
        discriminator.compile(loss='binary_crossentropy', optimizer=opt)
    else:
        discriminator.compile(loss=wasserstein_loss, optimizer=opt, metrics=['accuracy'])        
    return discriminator


def create_gan(generator, discriminator, opt=getGDopt(lr=0.01), loss='binary_crossentropy'):
    discriminator.trainable=False
    input_size = 100
    gan_input = Input(shape=(input_size,))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    
    if loss == 'binary_crossentropy':
        gan.compile(loss='binary_crossentropy', optimizer=opt)
    else:
        gan.compile(loss=wasserstein_loss, optimizer=opt, metrics=['accuracy'])        

    return gan

# gradient update steps for discriminator
def take_discriminator_steps(generator, discriminator, gan, X_train, k=1, NOISE_SIZE=100):
    for _ in range(k):
        noise= np.random.normal(0,1, [batch_size, NOISE_SIZE])
        generated_images = generator.predict(noise)

        image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]

        X= np.concatenate([image_batch, generated_images])

        y_dis = np.zeros(2*batch_size)
        y_dis[:batch_size] = 1

        discriminator.trainable=True
        loss = discriminator.train_on_batch(X, y_dis)

    return discriminator

# gradient update steps for discriminator
def take_discriminator_steps_wgan(generator, discriminator, gan, X_train, k=1, NOISE_SIZE=100, clip=False, clip_value=None):
    for _ in range(k):
        if clip and clip_value is not None:
            for layer in discriminator.layers:
                weights = layer.get_weights()
                weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                layer.set_weights(weights)        


        noise= np.random.normal(0,1, [batch_size, NOISE_SIZE])
#         generated_images = generator.predict(noise)

        image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]

#         X= np.concatenate([image_batch, generated_images])

#         y_dis = np.ones(2*batch_size)
#         y_dis[:batch_size] = -1

        positive_y = np.ones((batch_size, 1), dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((batch_size, 1), dtype=np.float32)
        
        discriminator.trainable=True
#         loss = discriminator.train_on_batch(X, y_dis)
        discriminator.train_on_batch([image_batch, noise],[positive_y, negative_y, dummy_y])
        #         print ("D-loss", loss)
        
    return discriminator


# perturbing weights of the generator
def perturb_generator(generator, sigma=0.001):
    weights, u = [], []
    for wt in generator.get_weights():
        u.append(np.random.normal(0, 1, wt.shape))
        wt = wt + u[-1] * sigma                
        weights.append(wt)
    
    generator.set_weights(weights)
        
    return generator

# gradient update steps for generator
def take_generator_steps(generator, discriminator, gan, NOISE_SIZE=100):
    noise= np.random.normal(0,1, [batch_size, NOISE_SIZE])
    generated_images = generator.predict(noise)

#     gan = create_gan(discriminator, generator)
    
    y_gen = np.ones(batch_size)
    discriminator.trainable=False
    gan.train_on_batch(noise, y_gen)
    
    return generator


def take_generator_steps_wgan(generator, discriminator, gan, NOISE_SIZE=100):
    noise= np.random.normal(0,1, [batch_size, NOISE_SIZE])
#     generated_images = generator.predict(noise)
    
#     y_gen = np.ones(batch_size)
    y_gen = np.ones((batch_size, 1), dtype=np.float32)
    discriminator.trainable=False
    loss = gan.train_on_batch(noise, y_gen)
#     print ("G-loss", loss)
    
    return generator


def change_network(modela, modelb):
    modela.set_weights(modelb.get_weights())
    return modela
    
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)

def getLoss(generator, discriminator, X_train, NOISE_SIZE=100):
    
    noise= np.random.normal(0,1, [batch_size, NOISE_SIZE])
    image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
    generated_images = generator.predict(noise)
    
    probabilities_1 = discriminator.predict(image_batch).reshape(len(image_batch))
    probabilities_1 = np.log(probabilities_1)
    
    probabilities_2 = discriminator.predict(generated_images).reshape(len(generated_images))
    probabilities_2 = np.log(1 - probabilities_2)

    return np.mean(probabilities_1) + np.mean(probabilities_2)


def getWassersteinLoss(generator, discriminator, X_train, NOISE_SIZE=100, eps=0):
    
    noise= np.random.normal(0,1, [batch_size, NOISE_SIZE])
    image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
    generated_images = generator.predict(noise)
    
    disc_1 = discriminator.predict(image_batch).reshape(len(image_batch))    
    disc_2 = discriminator.predict(generated_images).reshape(len(generated_images))
    y_pred = np.array(list(disc_1) + list(disc_2))
    y_true = np.ones(2*batch_size)
    y_true[:batch_size] = -1
    
    loss = np.mean(y_true*y_pred)
    return loss

def getLossFixedBatch(generator, discriminator, image_batch, generated_images):
    
    probabilities_1 = discriminator.predict(image_batch).reshape(len(image_batch))
    probabilities_1 = np.log(probabilities_1) / np.log(2)
    
    probabilities_2 = discriminator.predict(generated_images).reshape(len(generated_images))
    probabilities_2 = np.log(1 - probabilities_2) / np.log(2)

    return np.mean(probabilities_1) + np.mean(probabilities_2)

# Creating a GAN player object
def create_GAN_player():
    ganPlayer = Players(create_generator(), create_discriminator(), create_gan, take_generator_steps, take_discriminator_steps, change_network, change_network, perturb_generator)    
    return ganPlayer
    

def plot_generated_images(epoch, generator, folder="", save = False, image_shape=(28,28), examples=100, dim=(10,10), figsize=(10,10),name=""):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, *image_shape)
    generated_images = (generated_images + 1.0)/2.0
    plt.figure(figsize=figsize)
#     print(generated_images[0])
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig(folder+name %epoch)
    plt.close()

######################################################################




def generate_fake_FID_image_input(generator, image_shape=(28,28), examples=100, dim=(10,10), figsize=(10,10),name=""):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, *image_shape)
    generated_images = (generated_images + 1.0)/2.0
    return generated_images
    
###########    


# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]
	# generate class labels, -1 for 'real'
	y = -ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels with 1.0 for 'fake'
	y = ones((n_samples, 1))
	return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
	# prepare fake examples
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	# plot images
	for i in range(10 * 10):
		# define subplot
		plt.subplot(10, 10, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename1 = 'generated_plot_%04d.png' % (step+1)
	plt.savefig(filename1)
	plt.close()
	# save the generator model
	filename2 = 'model_%04d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
	# plot history
	plt.plot(d1_hist, label='crit_real')
	plt.plot(d2_hist, label='crit_fake')
	plt.plot(g_hist, label='gen')
	plt.legend()
	plt.savefig('plot_line_plot_loss.png')
	plt.close()

    

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
# 		print(new_image.shape)
# 		print(new_image)
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return mu1, mu2, sigma1, sigma2, fid


def scale_and_calculate_FID(model, images1, images2, new_shape=(299,299,3)):
    
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    
    mu1 = numpy.zeros((1, 2048))
    sigma1 = numpy.zeros((2048, 2048))
    for image in images1:
        images_list = list()
        image = resize(image, new_shape, 0)
        image = preprocess_input(image)
        images_list.append(image)
        act = model.predict(asarray(images_list))
        
        mu1 += act
        sigma1 += numpy.outer(act, act)
    n1 = float(images1.shape[0])
    mu1 /= n1
    sigma1 -= n1*numpy.outer(mu1, mu1)
    sigma1 /= (n1-1)

    mu2 = numpy.zeros((1,2048))
    sigma2 = numpy.zeros((2048, 2048))
    for image in images2:
        images_list = list()
        image = resize(image, new_shape, 0)
        image = preprocess_input(image)
        images_list.append(image)
        act = model.predict(asarray(images_list))
        
        mu2 += act
        sigma2 += numpy.outer(act, act)
    n2 = float(images2.shape[0])
    mu2 /= n2
    sigma2 -= n2*numpy.outer(mu2, mu2)
    sigma2 /= (n2-1)
    
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid

NOISE_SIZE = 100
IMAGE_SHAPE = (32,32,3)

def get_inception_score(model, player):
    is_score = []
    n, m = 10, 100
    for i in range(n):
        noise= np.random.normal(loc=0, scale=1, size=[m, 100])
        generated_images = player.get_x().predict(noise)
        generated_images = generated_images.reshape(m, *IMAGE_SHAPE)
        images = (generated_images + 1.0)/2.0

        images = images.astype('float32')
        images = scale_images(images, (299,299,3))
        eps=1E-16

        p_yx = model.predict(images)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        kl_d = p_yx * (np.log(p_yx) - np.log(p_y))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = np.mean(sum_kl_d)
        is_score.append(np.exp(avg_kl_d))
    return np.mean(is_score)

    