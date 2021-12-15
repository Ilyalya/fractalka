import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

im = Image.open('silver/large.jpg')


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [1/3, 1/3, 1/3])


imar = rgb2gray(np.array(im, dtype=np.int16))
sz = imar.shape


def fractal_signature(imar, d_):
    u = imar.copy()
    b = imar.copy()
    # ds = range(1, d_)

    footprint = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]])

    ds = range(4, 30)
    vols = []

    for d in ds:
        fst_u = u + 1
        fst_b = b - 1

        scnd_u = maximum_filter(u, mode='reflect', footprint=footprint, cval=0)
        scnd_b = minimum_filter(b, mode='reflect', footprint=footprint, cval=255)

        u = np.maximum(fst_u, scnd_u)
        b = np.minimum(fst_b, scnd_b)

        vols.append(np.sum(u - b))

    # plt.plot(ds, vols)
    # plt.show()
    return (vols[-1] - vols[-2]) / 2


# fractal_signature(imar, 100)

# eps = 100
# for start1, end1 in zip(range(0, imar.shape[0]-eps, eps), range(eps, imar.shape[0], eps)):
#     for start2, end2 in zip(range(0, imar.shape[1]-eps, eps), range(eps, imar.shape[1], eps)):
#         print(start1, end1)
#         print(start2, end2)
#         plt.imshow(imar[start1:end1, start2:end2])
#         # plt.show()

ass_d4 = []
ass_d10 = []
ass_d20 = []
ass_d50 = []
ass_d300 = []
epsilons = [10, 20, 30, 40, 50, 100, 200]
d_end = 15
for eps in epsilons:
    ads4 = ads10 = ads20 = ads50 = ads300 = 0
    for start1, end1 in zip(range(0, imar.shape[0]-eps, eps), range(eps, imar.shape[0], eps)):
        for start2, end2 in zip(range(0, imar.shape[1]-eps, eps), range(eps, imar.shape[1], eps)):
            # ads4 += fractal_signature(imar[start1:end1, start2:end2], 4)
            # ads10 += fractal_signature(imar[start1:end1, start2:end2], 10)
            # ads20 += fractal_signature(imar[start1:end1, start2:end2], 20)
            ads50 += fractal_signature(imar[start1:end1, start2:end2], d_end)
            # ads300 += fractal_signature(imar[start1:end1, start2:end2], 300)
    # ass_d4.append(ads4)
    # ass_d10.append(ads10)
    # ass_d20.append(ads20)
    ass_d50.append(ads50)
    # ass_d300.append(ads300)

# plt.plot(epses, ass_d4, label="d = 4")
# plt.plot(epses, ass_d10, label="d = 10")
# plt.plot(epses, ass_d20, label="d = 20")
plt.plot(epsilons, ass_d50)
# plt.plot(range(4, 30), ass_d300, label="d = 300")
plt.legend()
plt.show()
