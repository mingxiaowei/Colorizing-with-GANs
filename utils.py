import skimage as sk
import numpy as np
import pickle

def img_read(name):
    im = sk.img_as_float(sk.io.imread(name))
    if len(im.shape) == 3 and im.shape[2] == 4:
        im = sk.color.rgba2rgb(im)
    return im

def img_save(name, im):
    sk.io.imsave(name, sk.img_as_ubyte(im))

def img_val_clip(im):
    return np.clip(sk.img_as_float(im), 0, 1)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict