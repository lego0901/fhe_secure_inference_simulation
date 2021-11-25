import torch
import torch.nn.functional as F


# fmt: off
ABS_APPROX_CHEBYSHEV = [
    [], [], [], [],
    [0, 0.66667, 0, 0.33333],
    [-0.82843, 0, 1.8284, 0, 0],
    [0, -0.37771, 0, 1.1777, 0, 0.2],
    [1.4291, 0, -3.1197, 0, 2.6906, 0, 2.0817e-17],
    [0, 0.50247, 0, -1.3182, 0, 1.6729, 0, 0.14286],
    [-3.1826, 0, 8.2423, 0, -7.6223, 0, 3.5626, 0, 0],
    [0, -0.91275, 0, 2.6832, 0, -3.0449, 0, 2.1633, 0, 0.11111],
    [8.1093, 0, -24.768, 0, 28.298, 0, -15.078, 0, 4.4387, 0, -6.9389e-18],
    [0, 1.965, 0, -6.6171, 0, 8.6918, 0, -5.7824, 0, 2.6517, 0, 0.090909],
    [-22.469, 0, 79.462, 0, -109.74, 0, 74.655, 0, -26.227, 0, 5.317, 0, -6.0715e-17],
    [0, -4.7148, 0, 18.061, 0, -27.737, 0, 21.931, 0, -9.7554, 0, 3.139, 0, 0.076923],
    [65.93, 0, -265.46, 0, 431.58, 0, -362.32, 0, 166.87, 0, -41.808, 0, 6.1964, 0, -2.099e-16],
    [0, 12.201, 0, -52.574, 0, 93.157, 0, -87.622, 0, 47.335, 0, -15.189, 0, 3.6256, 0, 0.066667],
    [-201.71, 0, 911.73, 0, -1707.8, 0, 1712.1, 0, -990.13, 0, 332.31, 0, -62.561, 0, 7.0767, 0, -2.0817e-17],
    [0, -33.388, 0, 160.1, 0, -322.45, 0, 354.24, 0, -231.01, 0, 91.647, 0, -22.307, 0, 4.1119, 0, 0.058824],
    [637.07, 0, -3195.3, 0, 6779.7, 0, -7913, 0, 5532.5, 0, -2366, 0, 607.43, 0, -89.224, 0, 7.9574, 0, -4.5103e-17],
    [0, 95.388, 0, -504.2, 0, 1139, 0, -1435.3, 0, 1105, 0, -535.89, 0, 163.7, 0, -31.335, 0, 4.5978, 0, 0.052632],
]

ABS_APPROX_REMEZ = [  # 4 ~ 18
    [], [], [], [],
    [-1.0655, 3.1165e-15, 1.9303, -1.9508e-15, 0.067621],
    [-1.0655, -3.5928e-17, 1.9303, 3.0376e-17, 0.067621],
    [2.3103, 2.4256e-14, -4.1784, -2.5583e-14, 2.868, 5.2867e-15, 0.045929],
    [2.3103, -8.35e-16, -4.1784, 1.1087e-15, 2.868, -2.2213e-16, 0.045929],
    [-6.2356, 6.3699e-14, 13.72, -9.5696e-14, -10.363, 4.0129e-14, 3.8098, -4.0245e-15, 0.03469],
    [-6.2356, 2.821e-17, 13.72, -4.6771e-16, -10.363, 5.5024e-16, 3.8098, -8.8216e-17, 0.03469],
    [18.709, -7.109e-13, -49.592, 1.4493e-12, 47.775, -9.7486e-13, -20.646, 2.4019e-13, 4.7537, -1.5965e-14, 0.027845],
    [18.709, -8.499e-15, -49.592, 1.5114e-14, 47.775, -6.2749e-15, -20.646, -4.4504e-16, 4.7537, 6.53e-17, 0.027845],
    [-59.718, 2.92e-13, 186.41, -7.3989e-13, -222.43, 6.829e-13, 127.04, -2.8131e-13, -36.052, 5.0334e-14, 5.6986, -2.8612e-15, 0.023247],
    [-59.718, 3.4008e-14, 186.41, -9.4322e-14, -222.43, 9.4106e-14, 127.04, -4.0353e-14, -36.052, 6.7168e-15, 5.6986, -1.9982e-16, 0.023247],
    [198.46, -2.0874e-11, -714.9, 6.2957e-11, 1023, -7.2558e-11, -739.91, 3.982e-11, 285.33, -1.0515e-11, -57.607, 1.1732e-12, 6.6443, -3.6683e-14, 0.019949],
    [198.46, -7.9685e-14, -714.9, 2.1648e-13, 1023, -1.9506e-13, -739.91, 5.8174e-14, 285.33, 2.1156e-15, -57.607, -2.1377e-15, 6.6443, 1.3802e-16, 0.019949],
    [-678.46, 4.0175e-11, 2773.8, -1.42e-10, -4639.5, 2.0017e-10, 4085.9, -1.4327e-10, -2032, 5.4756e-11, 570, -1.0715e-11, -86.334, 9.2133e-13, 7.5904, -2.2567e-14, 0.017468],
    [-678.46, 1.1358e-13, 2773.8, -5.3243e-13, -4639.5, 9.5829e-13, 4085.9, -8.5086e-13, -2032, 3.9156e-13, 570, -8.7778e-14, -86.334, 7.9778e-15, 7.5904, -3.5549e-16, 0.017468],
    [2368.4, 5.0321e-10, -10842, -2.0179e-09, 20780, 3.3192e-09, -21627, -2.8839e-09, 13266, 1.4194e-09, -4872.3, -3.9368e-10, 1044.2, 5.7397e-11, -123.26, -3.7113e-12, 8.5369, 6.8856e-14, 0.015535],
]
# fmt: on


def abs_naive(x, *args, **kwargs):
    return x.abs()


def abs_chebyshev(x, n, clip_before=False):
    if clip_before:
        x = x.clip(-1, 1)

    xi, y = torch.ones(*x.shape).to(x.device), torch.zeros(*x.shape).to(x.device)
    l = len(ABS_APPROX_CHEBYSHEV[n])

    for i in range(l):
        y += ABS_APPROX_CHEBYSHEV[n][l - 1 - i] * xi
        xi *= x

    return y


def abs_remez(x, n, clip_before=True):
    if clip_before:
        x = x.clip(-1, 1)

    xi, y = torch.ones(*x.shape).to(x.device), torch.zeros(*x.shape).to(x.device)
    l = len(ABS_APPROX_REMEZ[n])

    for i in range(l):
        y += ABS_APPROX_REMEZ[n][l - 1 - i] * xi
        xi *= x

    return y


def abs_block(x, n, block_type="naive", clip_before=False):
    if block_type == "naive":
        return abs_naive(x, n)
    elif block_type == "chebyshev":
        return abs_chebyshev(x, n, clip_before=clip_before)
    elif block_type == "remez":
        return abs_remez(x, n, clip_before=clip_before)


def max_block(x, y, n, block_type="naive", clip_before=False):
    p, q = 0.5 * (x + y), 0.5 * abs_block(x - y, n, block_type, clip_before)
    return p + q


def relu_block(x, n, block_type="naive", clip_before=False):
    p = abs_block(x, n, block_type, clip_before)
    return 0.5 * (x + p)


def leaky_relu_block(x, n, negative_slope=0.1, block_type="naive", clip_before=False):
    p = abs_block(x, n, block_type, clip_before)
    return (
        (x + (1 - negative_slope) / (1 + negative_slope) * p)
        * (1 + negative_slope)
        * 0.5
    )


def maxpool2d(x, n=-1, block_type="naive", clip_before=False):
    # We only consider 2x2 kernel sized maxpool function for now
    # Also, only ceil_mode (11x11 -> 6x6) is supported
    max_function = lambda x, y: max_block(x, y, n, block_type, clip_before)
    row, col = x.shape[-2], x.shape[-1]

    if row != 0 or col != 0:
        # zero pad if the size is not divisible
        x = F.pad(x, (0, row % 2, 0, col % 2), "constant", 0)

    # Strided indexing
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    # 2x2 maxpool
    return max_function(max_function(x00, x01), max_function(x10, x11))


def relu(x, n=-1, block_type="naive", clip_before=False):
    return relu_block(x, n, block_type, clip_before)


def leaky_relu(x, n=-1, negative_slope=0.1, block_type="naive", clip_before=False):
    return leaky_relu_block(x, n, negative_slope, block_type, clip_before)
