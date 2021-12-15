import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage.filters import convolve
from scipy.stats import linregress

im = Image.open('silver/large.jpg')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [1/3, 1/3, 1/3])


def reni_entropy(p, q):
    return (1 / (1 - q) * np.log(np.sum(np.power(p, q)))) if q != 1 else (-np.sum(p * np.log(p)))

img = rgb2gray(np.array(im))
q = np.array(range(-2, 10))
ws = range(1, 20)
ns =[]
for w in ws:
    ns.append(reni_entropy(convolve(img, np.ones((w, w)), mode='constant')[::w, ::w] / np.mean(img), 10))

x = -np.log(ws)
y = ns

sns.regplot(x=pd.Series(x, name='log of window size (log(ϵ))'),
            y=pd.Series(y, name='N(q, ϵ)'))

linregress(x, y).slope
def get_reni_dim(img, q):
    ws = range(1, 20)
    ns = []

    for w in ws:
        conv = convolve(img, np.ones((w, w)), mode='constant')[::w, ::w]
        ns.append(reni_entropy(conv / np.sum(conv), q))

    x = -np.log(ws)
    y = ns

    return linregress(x, y).slope
def get_reni_spectre(img, qs):
    return list(map(lambda x: get_reni_dim(img, x), qs))
spec = get_reni_spectre(img, q)
plt.plot(q, spec)
plt.show()

ws = range(1, 20)
ns =[]
for w in ws:
    ns.append(reni_entropy(convolve(img, np.ones((w, w)), mode='constant')[::w, ::w] / np.mean(img), 1))
x = -np.log(ws)
y = ns
linregress(x, y).slope
reni_entropy(convolve(img, np.ones((w, w)), mode='constant')[::w, ::w] / np.mean(img), 1)