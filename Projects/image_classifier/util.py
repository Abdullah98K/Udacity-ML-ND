import tensorflow as tf
import numpy as np

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(images=image,size=[224,224])
    image = tf.cast(image, tf.float32)
    image /= 255
    return image.numpy()