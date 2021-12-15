import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage.filters import convolve, maximum_filter
from scipy.stats import linregress
from progressbar import ProgressBar

im = Image.open('silver/large.jpg')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [1/3, 1/3, 1/3])
immat = rgb2gray(np.array(im))

rs = range(2, 10)
dxs = []
for r in rs:
    dx = convolve(immat, np.ones((r*2, r*2)), mode='constant')
    dxs.append(dx.ravel())

frac_dims = []
bar = ProgressBar()
for l in bar(np.array(dxs).T):
    frac_dims.append(linregress(np.log(rs), np.log(l)).slope)

plt.imshow(np.array(frac_dims).reshape(immat.shape) >= np.mean(frac_dims)-0.3, cmap='gray')

plt.imshow(np.array(frac_dims).reshape(immat.shape) > np.mean(frac_dims)-0.6, cmap='gray')

def frac_dim(immat):
    ws = range(1, 10)
    ns = []
    for w in ws:
        ns.append(np.sum(maximum_filter(immat, (w, w), mode='constant')[::w, ::w]))

    x = np.log(1 / np.array(ws))
    y = np.log(ns)

    slope = linregress(x, y).slope

    return slope

frac_spec = []
alphas = np.linspace(np.min(frac_dims), np.max(frac_dims), num=15)
for a0, a1 in zip(alphas[:-1], alphas[1:]):
    frac_im = np.array(frac_dims).reshape(immat.shape)
    frac_im = (frac_im > a0) & (frac_im < a1)
    frac_spec.append(frac_dim(frac_im))

    plt.figure(figsize=(10, 10))
    plt.imshow(frac_im, cmap='Greys', interpolation='nearest')
    plt.show()

plt.plot(alphas[:-1], frac_spec, '-o')