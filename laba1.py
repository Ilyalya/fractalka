import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def fractal_dimension(Z, threshold=200):
    # Только для квадратного изображения
    assert (len(Z.shape) == 2)

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1)

        # Считаем непустые (0) и неполные ячейки (k * k)
        return len(np.where((S > 0) & (S < k * k))[0])

    # Преобразуем Z в двоичный массив
    Z = (Z < threshold)
    plt.imshow(Z)
    plt.show()

    # Минимальный размер изображения
    p = min(Z.shape)

    # Наибольшая степень 2 меньше или равна p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Извлечь экспоненту
    n = int(np.log(n) / np.log(2))

    # Построить последовательные размеры ячейки (от 2 ** n до 2 ** 1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Фактический подсчет ячеек с уменьшающимся размером
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Совместите последовательные log(размеры) с log (количество)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

    p = np.poly1d(coeffs)
    return -p[1]


I = rgb2gray(mpimg.imread("koh3.jpg"))
print("Размерность: ", fractal_dimension(I))
