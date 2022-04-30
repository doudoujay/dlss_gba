# Deep Learning Super Sampling
Using Deep Convolutional GANS to super sample images and increase their resolution.<br/>
![pokemon1](https://i.imgur.com/KT4mZPg.jpg)
![pokemon2](https://i.imgur.com/FZ66KOm.jpg)
![pokemon3](https://i.imgur.com/Sf1hnmt.jpg)

# How To Use This Repository
* ## [Tutorial](https://aix.web.tr/en/deep-learning-super-sampling-using-generative-adversarial-networks/)
* ## Requirements
  * Python 3
  * Keras (I use ```2.3.1```)
  * Tensorflow (I use ```1.14.0```)
  * Sklearn
  * Skimage
  * Numpy
  * Matplotlib
  * PIL
* ## Documentation
  * ## [DLSS GAN Training](https://nbviewer.jupyter.org/github/vee-upatising/DLSS/blob/master/DLSS%20GAN%20Training.ipynb)
      * This script is used to define the DCGAN class, train the Generative Adversarial Network, generate samples, and save the model at every epoch interval.
      * The Generator and Discriminator models were designed to be trained on an 8 GB GPU. If you have a less powerful GPU then decrease the ```conv_filter``` and ```kernel``` parameters accordingly.

      * ### User Specified Parameters:
          * ```input_path```: File path pointing to the folder containing the low resolution dataset.
          * ```output_path```: File path pointing to the folder containing the high resolution dataset.
          * ```input_dimensions```: Dimensions of the images inside the low resolution dataset. The image sizes must be compatible meaning ```output_dimensions / input_dimensions``` is a multiple of ```2```.
          * ```output_dimensions```: Dimensions of the images inside the high resolution dataset. The image sizes must be compatible meaning ```output_dimensions / input_dimensions``` is a multiple of ```2```.
          * ```super_sampling_ratio```: Integer representing the ratio of the difference in size between the two image resolutions. This integer specifies how many times the ```Upsampling2D``` and ```MaxPooling2D``` layers are used in the models.
          * ```model_path```: File path pointing to the folder where you want to save to model as well as generated samples.
          * ```interval```: Integer representing how many epochs between saving your model.
          * ```epochs```: Integer representing how many epochs to train the model.
          * ```batch```: Integer representing how many images to train at one time.
          * ```conv_filters```: Integer representing how many convolutional filters are used in each convolutional layer of the Generator and the Discriminator.
          * ```kernel```: Tuple representing the size of the kernels used in the convolutional layers.
          * ```png```: Boolean flag, set to True if the data has PNGs to remove alpha layer from images.

       * ### DCGAN Class:
          * ```__init__(self)```: The class is initialized by defining the dimensions of the input image as well as the output image. The Generator and Discriminator models get initialized using ```build_generator()``` and ```build_discriminator()```.
          * ```build_generator(self)```: Defines [Generator model](https://github.com/vee-upatising/DLSS/blob/master/README.md#generator-model-architecture). The ```Convolutional``` and ```UpSampling2D``` layers increase the resolution of the image by a factor of ```super_sampling_ratio * 2```. Gets called when the DCGAN class is initialized.
          * ```build_discriminator(self)```: Defines [Discriminator model](https://github.com/vee-upatising/DLSS/blob/master/README.md#discriminator-model-architecture). The ```Convolutional``` and ```MaxPooling2D``` layers downsample from ```output_dimensions``` to ```1``` scalar prediction. Gets called when the DCGAN class is initialized.
          * ```load_data(self)```: Loads data from user specified file path, ```data_path```. Reshapes images from ```input_path``` to have ```input_dimensions```. Reshapes  images from ```output_path``` to have ```output_dimensions```. Gets called in the ```train()``` method.
          * ```train(self, epochs, batch_size, save_interval)```: Trains the Generative Adversarial Network. Each epoch trains the model using the entire dataset split up into chunks defined by ```batch_size```. If epoch is at ```save_interval```, then the method calls ```save_imgs()``` to generate samples and saves the model at the current epoch.
          * ```save_imgs(self, epoch, gen_imgs, interpolated)```: Saves the model and [generates prediction samples](https://github.com/vee-upatising/DLSS/blob/master/README.md#generated-training-sample) for a given epoch at the user specified path, ```model_path```. Each sample contains 8 interpolated images and Deep Learned Super Sampled images for comparison.
          
  * ## [Load Model and Analyze Results](https://nbviewer.jupyter.org/github/vee-upatising/DLSS/blob/master/Load%20Model%20and%20Analyze%20Results.ipynb)
    * This script is used to super sample images using the Generator model trained by the ```DLSS GAN Training``` script.
    * The script will perform DLSS on all images inside the folder specified in ```dataset_path```. You can insert frames of a video in here to create GIFs such as the one in the [results](https://github.com/vee-upatising/DLSS/blob/master/README.md#results) section of this document.
    * ### User Specified Parameters:
        * ```input_dimensions```: Dimensions of the image resolution the model takes as input.
        * ```output_dimensions```: Dimensions of the image resolution the model takes as output.
        * ```super_sampling_ratio```: Integer representing the ratio of the difference in size between the two image resolutions. Used for setting ratio of image subplots.
        * ```model_path```: File path pointing to the folder where you want to save to model as well as generated samples.
        * ```dataset_path```: File path pointing to the folder containing dataset you want to perform DLSS on.
        * ```save_path```: File path pointing to the folder where you want to save generated predictions of the trained model.
        * ```png```: Boolean flag, set to True if the data has PNGs to remove alpha layer from images.


* ## Results
  * The results of [Nearest Neighbor Interpolation](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize) are compared to the DLSS algorithm. <br/>

![flip](https://vee-upatising.github.io/images/flip.gif)
![anime](https://vee-upatising.github.io/images/sr.jpg)
![comparison](https://raw.githubusercontent.com/vee-upatising/Super-Resolution-GAN/master/edited.png)

* ## Generated Training Sample
![Training](https://i.imgur.com/wCliEAM.png)

* ## Generator Model Architecture
  * Using ```(5,5)``` Convolutional Kernels with ```64``` filters.
  * ```input_dimensions = (128,128,3)``` and ```output_dimensions = (256,256,3)``` </br>
![Generator](https://i.imgur.com/Ll1UA4p.jpg)

* ## Discriminator Model Architecture
  * Using ```(5,5)``` Convolutional Kernels with ```64``` filters.
  * ```input_dimensions = (128,128,3)``` and ```output_dimensions = (256,256,3)``` </br>
![Discriminator](https://i.imgur.com/Pi8gTJR.jpg)
