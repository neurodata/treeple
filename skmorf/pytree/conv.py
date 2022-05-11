import numpy as np
import torch


def _apply_convolution(sample_X, kernel, image_height, image_width):
    """Apply convolution to a vectorized data point with a specified kernel.

    Parameters
    ----------
    sample_X : [type]
        [description]
    kernel : [type]
        [description]
    image_height : [type]
        [description]
    image_width : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # reshape into a pytorch tensor
    sample_X = torch.tensor(sample_X.reshape(-1, 1, image_height, image_width))
    tensor_kernel = kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1])

    # extract the real and imaginary kernels
    real_kernel = tensor_kernel.real
    imag_kernel = tensor_kernel.imag

    # create a (2, 1, K_h, K_w) tensor with 2
    tensor_kernel = np.concatenate((real_kernel, imag_kernel), axis=0)
    tensor_kernel = torch.tensor(tensor_kernel)

    # how much to pad by to do 'same' padding
    pad_size = list(map(int, ((kernel.shape[0] - 1) / 2, (kernel.shape[1] - 1) / 2)))

    # apply convolution
    output = torch.conv2d(sample_X, tensor_kernel, padding=pad_size)
    return output
