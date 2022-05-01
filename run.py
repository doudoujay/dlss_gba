from PIL import Image
from pyboy import PyBoy
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Dimensions of the images inside the dataset.
input_dimensions = (128, 128, 3)
# Dimensions of the images inside the dataset.
output_dimensions = (256, 256, 3)
# The ratio of the difference in size of the two images. Used for setting ratio of image subplots
super_sampling_ratio = int(output_dimensions[0] / input_dimensions[0])
# Path to saved .h5 model
model_path = './DLSS/generator.h5'
# Boolean flag, set to True if the data has pngs to remove alpha layer from images
png = True

def upscale(img: Image, model, counter):
    # original image
    x = img.resize((input_dimensions[0], input_dimensions[1]))

    # interpolated (resized) image
    y = x.resize((output_dimensions[0], output_dimensions[1]))

    x = np.array(x)
    y = np.array(y)

    # Remove alpha layer if imgaes are PNG
    if(png):
        x = x[..., :3]
        y = y[..., :3]

    # plotting super sampled image
    x = x.reshape(1, input_dimensions[0],
                  input_dimensions[1], input_dimensions[2])/255
    result = np.array(model.predict_on_batch(x))*255
    result = result.reshape(
        output_dimensions[0], output_dimensions[1], output_dimensions[2])
    np.clip(result, 0, 255, out=result)
    result = result.astype('uint8')
    # print(result)
    # plt.imshow(result)
    im = Image.fromarray(result)
    im.save(f'./frames/{counter}.png')

model = load_model(model_path)
pyboy = PyBoy('rom/red.gb')
# while not pyboy.tick():
#     pil_image = pyboy.screen_image()
#     print(pil_image)
#     pass
counter = 0
while not pyboy.tick():
    pil_image = pyboy.screen_image()
    upscale(pil_image, model, counter)
    counter += 1
