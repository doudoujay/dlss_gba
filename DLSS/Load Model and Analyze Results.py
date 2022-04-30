#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.models import load_model


# # Parameters

# In[21]:


# Dimensions of the images inside the dataset.
input_dimensions = (128,128,3)

# Dimensions of the images inside the dataset.
output_dimensions = (256,256,3)

# The ratio of the difference in size of the two images. Used for setting ratio of image subplots
super_sampling_ratio = int(output_dimensions[0] / input_dimensions[0])

# Path to saved .h5 model
model_path = r'C:\Users\Vee\Desktop\generator90.h5'

# Path to folder containing images to super sample
dataset_path = r'C:\Users\Vee\Desktop\pokemon'

# Folder where you want to save to model as well as generated samples
save_path = r'C:\Users\Vee\Desktop'

# Boolean flag, set to True if the data has pngs to remove alpha layer from images
png = True


# # Load Model

# In[19]:


model = load_model(model_path)


# # Load Images and Super Sample

# In[29]:


paths = []
count = 0

for r, d, f in os.walk(dataset_path):
    for file in f:
        if '.png' in file or 'jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    
    # Select image
    img = Image.open(path)

    #create plot
    f, axarr = plt.subplots(1,3,figsize=(15,15),gridspec_kw={'width_ratios': [1,super_sampling_ratio,super_sampling_ratio]})
    axarr[0].set_xlabel('Original Image (' + str(input_dimensions[0]) + 'x' + str(input_dimensions[1]) + ')', fontsize=10)
    axarr[1].set_xlabel('Interpolated Image (' + str(output_dimensions[0]) + 'x' + str(output_dimensions[1]) + ')', fontsize=10)
    axarr[2].set_xlabel('Super Sampled Image (' + str(output_dimensions[0]) + 'x' + str(output_dimensions[1]) + ')', fontsize=10)

    #original image
    x = img.resize((input_dimensions[0],input_dimensions[1]))
    
    #interpolated (resized) image
    y = x.resize((output_dimensions[0],output_dimensions[1]))
    
    
    x = np.array(x)
    y = np.array(y)
    
    # Remove alpha layer if imgaes are PNG
    if(png):
        x = x[...,:3]
        y = y[...,:3]
    
    #plotting first two images
    axarr[0].imshow(x)
    axarr[1].imshow(y)
    
    #plotting super sampled image
    x = x.reshape(1,input_dimensions[0],input_dimensions[1],input_dimensions[2])/255
    result = np.array(model.predict_on_batch(x))*255
    result = result.reshape(output_dimensions[0],output_dimensions[1],output_dimensions[2])
    np.clip(result, 0, 255, out=result)
    result = result.astype('uint8')
                
    axarr[2].imshow(result)
    
    # Save image
    f.savefig(save_path + '\\frame_%d.png' % count)
    
    # Increment file name counter
    count = count + 1

