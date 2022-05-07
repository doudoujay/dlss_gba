#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import skimage.transform as st


# # Set Parameters

# In[2]:


# Folder containing input (low resolution) dataset
input_path = r'D:\Downloads\selfie2anime\trainB'

# Folder containing output (high resolution) dataset
output_path = r'D:\Downloads\selfie2anime\trainB'

# Dimensions of the images inside the dataset.
# NOTE: The image sizes must be compatible meaning output_dimensions / input_dimensions is a multiple of 2
input_dimensions = (128,128,3)

# Dimensions of the images inside the dataset.
# NOTE: The image sizes must be compatible meaning output_dimensions / input_dimensions is a multiple of 2
output_dimensions = (256,256,3)

# How many times to increase the resolution by 2 (by appling the UpSampling2D layer)
super_sampling_ratio = int(output_dimensions[0] / input_dimensions[0] / 2)

# Folder where you want to save to model as well as generated samples
model_path = r"C:\Users\Vee\Desktop\python\GAN\DLSS\results"

# How many epochs between saving your model
interval = 5

# How many epochs to train the model
epoch = 100

# How many images to train at one time. Ideally this number would be a factor of the size of your dataset
batch = 25

# How many convolutional filters for each convolutional layer of the generator and the discrminator
conv_filters = 64

# Size of kernel used in the convolutional layers
kernel = (5,5)

# Boolean flag, set to True if the data has pngs to remove alpha layer from images
png = True


# # Create Deep Convolutional GAN Class

# In[3]:


class DCGAN():
    
    # Initialize parameters, generator, and discriminator models
    def __init__(self):
        
        # Set dimensions of the output image
        self.img_rows = output_dimensions[0]
        self.img_cols = output_dimensions[1]
        self.channels = output_dimensions[2]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        # Shape of low resolution input image
        self.latent_dim = input_dimensions
        
        # Chose optimizer for the models
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        generator = self.generator

        # The generator takes low resolution images as input and generates high resolution images
        z = Input(shape = self.latent_dim)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    # load data from specified file path 
    def load_data(self):
        
        # Initializing arrays for data and image file paths
        data = []
        small = []
        paths = []
        
        # Get the file paths of all image files in this folder
        for r, d, f in os.walk(output_path):
            for file in f:
                if '.jpg' in file or 'png' in file:
                    paths.append(os.path.join(r, file))
                    
        # For each file add high resolution image to array
        for path in paths:
            img = Image.open(path)
            
            # Resize Image
            y = np.array(img.resize((self.img_rows,self.img_cols)))
            
            # Remove alpha layer if imgaes are PNG
            if(png):
                y = y[...,:3]
                
            data.append(y)
          
        paths = []
        
        # Get the file paths of all image files in this folder
        for r, d, f in os.walk(input_path):
            for file in f:
                if '.jpg' in file or 'png' in file:
                    paths.append(os.path.join(r, file))
                    
        # For each file add low resolution image to array
        for path in paths:
            img = Image.open(path)
            
            # Resize Image
            x = np.array(img.resize((self.latent_dim[0],self.latent_dim[1])))
            
            # Remove alpha layer if imgaes are PNG
            if(png):
                x = x[...,:3]
                
            small.append(x)
        
            
        # Return x_train and y_train reshaped to 4 dimensions
        y_train = np.array(data)
        y_train = y_train.reshape(len(data),self.img_rows,self.img_cols,self.channels)
        x_train = np.array(small)
        x_train = x_train.reshape(len(small),self.latent_dim[0],self.latent_dim[1],self.latent_dim[2])
        
        del data
        del small
        del paths
        
        # Shuffle indexes of data
        X_shuffle, Y_shuffle = shuffle(x_train, y_train)
        
        return X_shuffle, Y_shuffle
    
    # Define Generator model
    def build_generator(self):

        model = Sequential()
        
        # 1st Convolutional Layer / Input Layer
        model.add(Conv2D(conv_filters, kernel_size=kernel, padding="same", input_shape=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        
        # Upsample the data as many times as needed to reach output resolution
        for i in range(super_sampling_ratio):
        
            # Super Sampling Convolutional Layer
            model.add(Conv2D(conv_filters, kernel_size=kernel, padding="same"))
            model.add(LeakyReLU(alpha=0.2))

            # Upsample the data (Double the resolution)
            model.add(UpSampling2D())

        # Convolutional Layer
        model.add(Conv2D(conv_filters, kernel_size=kernel, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        # Convolutional Layer
        model.add(Conv2D(conv_filters, kernel_size=kernel, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        
        # Final Convolutional Layer (Output Layer)
        model.add(Conv2D(3, kernel_size=kernel, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.summary()

        noise = Input(shape=self.latent_dim)
        img = model(noise)

        return Model(noise, img)

    # Define Discriminator model
    def build_discriminator(self):

        model = Sequential()

        # Input Layer
        model.add(Conv2D(conv_filters, kernel_size=kernel, input_shape=self.img_shape,activation = "relu", padding="same"))
        
        # Downsample the image as many times as needed
        for i in range(super_sampling_ratio):
            
            # Convolutional Layer
            model.add(Conv2D(conv_filters, kernel_size=kernel))
            model.add(LeakyReLU(alpha=0.2))
        
            # Downsample the data (Half the resolution)
            model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Convolutional Layer
        model.add(Conv2D(conv_filters, kernel_size=kernel, strides = 2))
        model.add(LeakyReLU(alpha=0.2))

        # Convolutional Layer
        model.add(Conv2D(conv_filters, kernel_size=kernel, strides = 2))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Flatten())
        
        # Output Layer
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
    
    # Train the Generative Adversarial Network
    def train(self, epochs, batch_size, save_interval):
        
        # Prevent script from crashing from bad user input
        if(epochs <= 0):
            epochs = 1
        
        if(batch_size <= 0):
            batch_size = 1

        # Load the dataset
        X_train, Y_train = self.load_data()
        
        # Normalizing data to be between 0 and 1
        X_train = X_train / 255
        Y_train = Y_train / 255

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Placeholder arrays for Loss function values
        g_loss_epochs = np.zeros((epochs, 1))
        d_loss_epochs = np.zeros((epochs, 1))
        
        # Training the GAN
        for epoch in range(1, epochs + 1):
            
            # Initialize indexes for training data
            start = 0
            end = start + batch_size
            
            # Array to sum up all loss function values
            discriminator_loss_real = []
            discriminator_loss_fake = []
            generator_loss = []
            
            # Iterate through dataset training one batch at a time
            for i in range(int(len(X_train)/batch_size)):
                
                # Get batch of images
                imgs_output = Y_train[start:end]
                imgs_input = X_train[start:end]

                # Train Discriminator

                # Make predictions on current batch using generator
                gen_imgs = self.generator.predict(imgs_input)

                # Train the discriminator (real classified as ones and generated as zero)
                d_loss_real = self.discriminator.train_on_batch(imgs_output, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                #  Train Generator

                # Train the generator (wants discriminator to mistake images as real)
                g_loss = self.combined.train_on_batch(imgs_input, valid)
                
                # Add loss for current batch to sum over entire epoch
                discriminator_loss_real.append(d_loss[0])
                discriminator_loss_fake.append(d_loss[1])
                generator_loss.append(g_loss)
                
                # Increment image indexes
                start = start + batch_size
                end = end + batch_size
             
            
            # Get average loss over the entire epoch
            loss_data = [np.average(discriminator_loss_real),np.average(discriminator_loss_fake),np.average(generator_loss)]
            
            #save loss history
            g_loss_epochs[epoch - 1] = loss_data[2]
            
            # Average loss of real data classification and fake data accuracy
            d_loss_epochs[epoch - 1] = (loss_data[0] + (1 - loss_data[1])) / 2
                
            # Print average loss over current epoch
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, loss_data[0], loss_data[1]*100, loss_data[2]))

            # If epoch is at interval, save model and generate image samples
            if epoch % save_interval == 0:
                
                # Select 8 random indexes
                idx = np.random.randint(0, X_train.shape[0], 8)
                
                # Get batch of training images
                x_points = X_train[idx]
                
                # Make predictions on batch of training images
                predicted_imgs = self.generator.predict(x_points)
                
                # Undo normalization of data. Update values to be between 0 and 255 for RGB image
                predicted_imgs = np.array(predicted_imgs) * 255
                np.clip(predicted_imgs, 0, 255, out=predicted_imgs)
                predicted_imgs = predicted_imgs.astype('uint8')
                x_points = np.array(x_points) * 255
                np.clip(x_points, 0, 255, out=x_points)
                x_points = x_points.astype('uint8')
                
                interpolated_imgs = []
                
                # Interpolate low resolution images for comparison
                for x in range(len(x_points)):
                    img = Image.fromarray(x_points[x])
                    interpolated_imgs.append(np.array(img.resize((self.img_rows,self.img_cols))))
                
                # Plot the predictions next to the interpolated images
                self.save_imgs(epoch, predicted_imgs, interpolated_imgs)
                
        return g_loss_epochs, d_loss_epochs
    
    # Save the model and generate prediction samples for a given epoch
    def save_imgs(self, epoch, gen_imgs, interpolated):
        
        # Define number of columns and rows
        r, c = 4, 4
        
        # Placeholder array for MatPlotLib Figure Subplots
        subplots = []
        
        # Create figure with title
        fig = plt.figure(figsize= (40, 40))
        fig.suptitle("Epoch: " + str(epoch), fontsize=65)
        
        # Initialize counters needed to track indexes across multiple arrays
        img_count = 0;
        index_count = 0;
        x_count = 0;
        
        # Loop through columns and rows of the figure
        for i in range(1, c+1):
            for j in range(1, r+1):
                # If row is even, plot the predictions
                if(j % 2 == 0):
                    img = gen_imgs[index_count]
                    index_count = index_count + 1
                # If row is odd, plot the interpolated images
                else:
                    img = interpolated[x_count]
                    x_count = x_count + 1
                # Add image to figure, add subplot to array
                subplots.append(fig.add_subplot(r, c, img_count + 1))
                plt.imshow(img)
                img_count = img_count + 1
        
        # Add title to columns of figure
        subplots[0].set_title("Interpolated", fontsize=45)
        subplots[1].set_title("Predicted", fontsize=45)
        subplots[2].set_title("Interpolated", fontsize=45)
        subplots[3].set_title("Predicted", fontsize=45)
                
        # Save figure to .png image in specified folder
        fig.savefig(model_path + "\\epoch_%d.png" % epoch)
        plt.close()
        
        # save model to .h5 file in specified folder
        self.generator.save(model_path + "\\generator" + str(epoch) + ".h5")


# # Initialize Generator and Discriminator Models

# In[4]:


dcgan = DCGAN()


# # Train GAN

# In[5]:


g_loss, d_loss = dcgan.train(epochs=epoch, batch_size=batch, save_interval=interval)


# # Plot Loss

# In[6]:


plt.plot(g_loss)
plt.plot(d_loss)
plt.title('GAN Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Generator', 'Discriminator'], loc='upper left')
plt.show()

