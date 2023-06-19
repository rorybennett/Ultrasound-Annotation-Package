import numpy
from numba import jit


@jit(nopython=True)
def checkCurrentLabels(labels_count, current_labels):
    for i in range(4):
        if current_labels[i] == 0:
            return False
        if labels_count[i][current_labels[i]] == labels_count[i][0]:
            return False
    return True


@jit(nopython=True)
def checkLabels(labels_count):
    good_count = 0
    for i in range(4):
        good_count_2 = 0
        for j in range(len(labels_count[i])):
            if labels_count[i][0] * 2 >= labels_count[i][j] and labels_count[i][0] <= labels_count[i][j]:
                good_count_2 += 1
        if good_count_2 == len(labels_count[i]):
            good_count += 1
    if good_count == 4:
        return False
    else:
        return True


@jit(nopython=True)
def get_label(val, intervals):
    i = -1
    for interval in intervals:
        i += 1
        if interval[0] <= val < interval[1]:
            return i


@jit(nopython=True)
def getAngle(p1, p2):
    a = numpy.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / numpy.pi
    if a < 0:
        a = 360 + a
    return a


# Create and return a patch of the given coordinate from the given image

@jit(nopython=True)
def createPatch(image, x, y, patch_size):
    row = y
    col = x

    patch = numpy.zeros((patch_size, patch_size))
    pr = -1
    for r in range(row - patch_size // 2, row + patch_size // 2):
        pr += 1
        pc = -1
        for c in range(col - patch_size // 2, col + patch_size // 2):
            pc += 1
            if r < 0 or r >= image.shape[0] or c < 0 or c >= image.shape[1]:
                patch[pr][pc] = 0
            else:
                patch[pr][pc] = image[r][c]

    return patch


@jit(nopython=True)
def concat(arrPatches):
    return numpy.concatenate((numpy.concatenate((arrPatches[0], arrPatches[1]), 1),
                              numpy.concatenate((arrPatches[2], arrPatches[3]), 1)), 0)


ang_val = getAngle((10, 10), (5, 15))
