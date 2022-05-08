import itertools
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import skimage.transform as st
from constants import *


class DCGAN():

    def __init__(self):

        self.img_rows = output_dimensions[0]
        self.img_cols = output_dimensions[1]
        self.channels = output_dimensions[2]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.latent_dim = input_dimensions

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()
        generator = self.generator

        z = Input(shape=self.latent_dim)
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def load_data(self):

        data = []
        small = []
        paths = []

        for r, d, f in os.walk(output_path):
            paths.extend(os.path.join(r, file)
                         for file in f if '.jpg' in file or 'png' in file)

        for path in paths:
            img = Image.open(path)

            y = np.array(img.resize((self.img_rows, self.img_cols)))

            if(png):
                y = y[..., :3]

            data.append(y)

        paths = []

        for r, d, f in os.walk(input_path):
            paths.extend(os.path.join(r, file)
                         for file in f if '.jpg' in file or 'png' in file)

        for path in paths:
            img = Image.open(path)

            x = np.array(img.resize((self.latent_dim[0], self.latent_dim[1])))

            if(png):
                x = x[..., :3]

            small.append(x)

        y_train = np.array(data)
        y_train = y_train.reshape(
            len(data), self.img_rows, self.img_cols, self.channels)
        x_train = np.array(small)
        x_train = x_train.reshape(
            len(small), self.latent_dim[0], self.latent_dim[1], self.latent_dim[2])

        del data
        del small
        del paths

        X_shuffle, Y_shuffle = shuffle(x_train, y_train)

        return X_shuffle, Y_shuffle

    def build_generator(self):

        model = Sequential()

        model.add(Conv2D(conv_filters, kernel_size=kernel,
                  padding="same", input_shape=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))

        for _ in range(super_sampling_ratio):
            model.add(Conv2D(conv_filters, kernel_size=kernel, padding="same"))
            model.add(LeakyReLU(alpha=0.2))

            model.add(UpSampling2D())

        model.add(Conv2D(conv_filters, kernel_size=kernel, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(conv_filters, kernel_size=kernel, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(3, kernel_size=kernel, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.summary()

        noise = Input(shape=self.latent_dim)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(conv_filters, kernel_size=kernel,
                  input_shape=self.img_shape, activation="relu", padding="same"))

        for _ in range(super_sampling_ratio):
            model.add(Conv2D(conv_filters, kernel_size=kernel))
            model.add(LeakyReLU(alpha=0.2))

            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(conv_filters, kernel_size=kernel, strides=2))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(conv_filters, kernel_size=kernel, strides=2))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())

        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size, save_interval):

        if(epochs <= 0):
            epochs = 1

        if(batch_size <= 0):
            batch_size = 1

        X_train, Y_train = self.load_data()

        X_train = X_train / 255
        Y_train = Y_train / 255

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        g_loss_epochs = np.zeros((epochs, 1))
        d_loss_epochs = np.zeros((epochs, 1))

        for epoch in range(1, epochs + 1):

            start = 0
            end = start + batch_size

            discriminator_loss_real = []
            discriminator_loss_fake = []
            generator_loss = []

            for _ in range(int(len(X_train)/batch_size)):
                imgs_output = Y_train[start:end]
                imgs_input = X_train[start:end]

                gen_imgs = self.generator.predict(imgs_input)

                d_loss_real = self.discriminator.train_on_batch(
                    imgs_output, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                g_loss = self.combined.train_on_batch(imgs_input, valid)

                discriminator_loss_real.append(d_loss[0])
                discriminator_loss_fake.append(d_loss[1])
                generator_loss.append(g_loss)

                start = start + batch_size
                end = end + batch_size

            loss_data = [np.average(discriminator_loss_real), np.average(
                discriminator_loss_fake), np.average(generator_loss)]

            g_loss_epochs[epoch - 1] = loss_data[2]

            d_loss_epochs[epoch - 1] = (loss_data[0] + (1 - loss_data[1])) / 2

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (epoch, loss_data[0], loss_data[1]*100, loss_data[2]))

            if epoch % save_interval == 0:

                idx = np.random.randint(0, X_train.shape[0], 8)

                x_points = X_train[idx]

                predicted_imgs = self.generator.predict(x_points)

                predicted_imgs = np.array(predicted_imgs) * 255
                np.clip(predicted_imgs, 0, 255, out=predicted_imgs)
                predicted_imgs = predicted_imgs.astype('uint8')
                x_points = np.array(x_points) * 255
                np.clip(x_points, 0, 255, out=x_points)
                x_points = x_points.astype('uint8')

                interpolated_imgs = []

                for x in range(len(x_points)):
                    img = Image.fromarray(x_points[x])
                    interpolated_imgs.append(
                        np.array(img.resize((self.img_rows, self.img_cols))))

                self.save_imgs(epoch, predicted_imgs, interpolated_imgs)

        return g_loss_epochs, d_loss_epochs

    def save_imgs(self, epoch, gen_imgs, interpolated):

        r, c = 4, 4

        subplots = []

        fig = plt.figure(figsize=(40, 40))
        fig.suptitle(f"Epoch: {str(epoch)}", fontsize=65)

        img_count = 0
        index_count = 0
        x_count = 0

        for _, j in itertools.product(range(1, c+1), range(1, r+1)):
            if(j % 2 == 0):
                img = gen_imgs[index_count]
                index_count = index_count + 1

            else:
                img = interpolated[x_count]
                x_count = x_count + 1

            subplots.append(fig.add_subplot(r, c, img_count + 1))
            plt.imshow(img)
            img_count = img_count + 1

        subplots[0].set_title("Interpolated", fontsize=45)
        subplots[1].set_title("Predicted", fontsize=45)
        subplots[2].set_title("Interpolated", fontsize=45)
        subplots[3].set_title("Predicted", fontsize=45)

        fig.savefig(model_path + "\\epoch_%d.png" % epoch)
        plt.close()

        self.generator.save(model_path + "\\generator" + str(epoch) + ".h5")


if __name__ == "__main__":
    dcgan = DCGAN()
    g_loss, d_loss = dcgan.train(
        epochs=epoch, batch_size=batch, save_interval=interval)
    plt.plot(g_loss)
    plt.plot(d_loss)
    plt.title('GAN Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Generator', 'Discriminator'], loc='upper left')
    plt.show()
