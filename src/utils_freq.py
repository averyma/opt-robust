import torch
import os
import numpy as np
from scipy import signal

def rgb2gray(rgb_input):
    """
        reference: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    """
    # batch operation
    if len(rgb_input.shape) == 4:
        gray_output = rgb_input[:, 0, :, :]*0.299 + \
                        rgb_input[:, 1, :, :]*0.587 + \
                        rgb_input[:, 2, :, :]*0.114
    # single image operation
    elif len(rgb_input.shape) == 3:
        gray_output = rgb_input[0, :, :]*0.299 + \
                        rgb_input[1, :, :]*0.587 + \
                        rgb_input[2, :, :]*0.114
 
    else:
        raise  NotImplementedError("Input dimension not supported. Check tensor shape!")
 
    return gray_output

def getDCTmatrix(size):
    home_dir = '/scratch/ssd001/home/ama/workspace/opt-robust/dct_matrix/'
    """
        Computed using C_{jk}^{N} found in the following link:
        https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
        Verified with cv2.dct(), error less than 1.1260e-06.

        output: DCT matrix with shape (size,size)
    """
    dct_matrix = torch.zeros([size, size])
    if size in [28, 32, 224, 784]:
        dct_matrix_path = os.path.join(home_dir, str(size) + '.pt')
        dct_matrix = torch.load(dct_matrix_path)
    else:
        for i in range(0, size):
            for j in range(0, size):
                if j == 0:
                    dct_matrix[i, j] = np.sqrt(1/size)*np.cos(np.pi*(2*i+1)*j/2/size)
                else:
                    dct_matrix[i, j] = np.sqrt(2/size)*np.cos(np.pi*(2*i+1)*j/2/size)

    return dct_matrix

def dct(input_tensor):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    size = input_tensor.shape[0]

    dct_matrix = getDCTmatrix(size).to(input_tensor.device)
    dct_output = torch.mm(dct_matrix.transpose(0, 1), input_tensor)

    return dct_output

def batch_dct(input_tensor, dct_matrix):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    m = input_tensor.shape[0]
    d = input_tensor.shape[1]

    dct_matrix = dct_matrix.to(input_tensor.device).expand(m,-1,-1)
    dct_output = torch.bmm(dct_matrix.transpose(1, 2), input_tensor.view(m,d,1)).squeeze()
    
    return dct_output

def batch_idct(input_tensor, dct_matrix):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    m = input_tensor.shape[0]
    d = input_tensor.shape[1]

    idct_matrix = torch.inverse(dct_matrix).to(input_tensor.device).expand(m,-1,-1)
    idct_output = torch.bmm(idct_matrix.transpose(1, 2), input_tensor.view(m,d,1)).squeeze()
    
    return idct_output

def idct(input_tensor):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    size = input_tensor.shape[0]
    dct_matrix = getDCTmatrix(size).to(input_tensor.device)
    idct_matrix = torch.inverse(dct_matrix)
    idct_output = torch.mm(idct_matrix.transpose(0, 1), input_tensor)

    return idct_output

def dct2(input_tensor):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
        
        Note that this operation is performed on a single image
        input shape: (size,size)
        output shape: (size, size)
    """ 
    
    size = input_tensor.shape[0]
    dct_matrix = getDCTmatrix(size).to(input_tensor.device)
    dct_output = torch.mm(torch.mm(dct_matrix.transpose(0, 1), input_tensor),dct_matrix)

    return dct_output

def idct2(input_tensor):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
        
        Note that this operation is performed on a single image
        input shape: (size,size)
        output shape: (size, size)
    """ 
    size = input_tensor.shape[0]
    idct_matrix = torch.inverse(getDCTmatrix(size)).to(input_tensor.device)
    idct_output = torch.mm(torch.mm(idct_matrix.transpose(0, 1), input_tensor.squeeze()),idct_matrix)

    return idct_output

def batch_dct2(input_tensor, dct_matrix):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    batch_size = input_tensor.shape[0]
    d = input_tensor.shape[2]

    dct_matrix = dct_matrix.to(input_tensor.device).expand(batch_size,-1,-1)
    dct2_output = torch.bmm(torch.bmm(dct_matrix.transpose(1, 2), input_tensor.view(batch_size,d,d)), dct_matrix)
    
    return dct2_output

def batch_idct2(input_tensor, dct_matrix):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    batch_size = input_tensor.shape[0]
    d = input_tensor.shape[2]

    idct_matrix = torch.inverse(dct_matrix).to(input_tensor.device).expand(batch_size,-1,-1)
    idct2_output = torch.bmm(torch.bmm(idct_matrix.transpose(1, 2), input_tensor.view(batch_size,d,d)), idct_matrix)
    
    return idct2_output

def batch_dct2_3channel(input_tensor, dct_matrix):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    assert len(input_tensor.shape) == 4 and input_tensor.shape[1]==3
    batch_size = input_tensor.shape[0]
    dct2_output = torch.zeros_like(input_tensor, device = input_tensor.device)
    d = input_tensor.shape[2]
    dct_matrix = dct_matrix.to(input_tensor.device).expand(batch_size,-1,-1)

    for i in range(3):
        dct2_output[:,i,:,:] = torch.bmm(torch.bmm(dct_matrix.transpose(1, 2), input_tensor[:,i,:,:].view(batch_size,d,d)), dct_matrix)
    
    return dct2_output

def batch_idct2_3channel(input_tensor, dct_matrix):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    assert len(input_tensor.shape) == 4 and input_tensor.shape[1]==3
    batch_size = input_tensor.shape[0]
    idct2_output = torch.zeros_like(input_tensor, device = input_tensor.device)
    d = input_tensor.shape[2]
    idct_matrix = torch.inverse(dct_matrix).to(input_tensor.device).expand(batch_size,-1,-1)
    
    for i in range(3):
        idct2_output[:,i,:,:] = torch.bmm(torch.bmm(idct_matrix.transpose(1, 2), input_tensor[:,i,:,:].view(batch_size,d,d)), idct_matrix)
    
    return idct2_output


def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0
    
def distance_from_top_left(i, j):
    dis = np.sqrt((i) ** 2 + (j) ** 2)
    return dis
#     if dis < r:
#         return 1.0
#     else:
#         return 0

# this generates a binary mask which sqrt(i^2+j^2)<r is 1
def mask_radial(size, r):
#     rows, cols = img.shape
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            mask[i, j] = distance_from_top_left(i, j) < r
    return mask

# this generates a binary mask which sqrt(i^2+j^2)<r is 1
def mask_radial_multiple_radius(size, r_list):
#     rows, cols = img.shape
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            flag = False
            for k,r in enumerate(r_list):
                if distance_from_top_left(i, j) < r:
                    mask[i, j] = k
                    flag = True
                    break
            if not flag:
                mask[i,j] = len(r_list)
                    
    return mask

# this generates a binary mask which (sqrt(i^2+j^2)-r).abs()<threshold is 1
def equal_dist_from_top_left(size, r):
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            dis = np.sqrt((i) ** 2 + (j) ** 2)
            if np.abs(dis-r) < 0.5:
                mask[i, j] = 1.0
            else:
                mask[i, j] = 0
    return mask


def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n] = data.shape
    mask = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                mask[i,j] = 1
            else:
                mask[i,j] = r
    mask = mask
    for i in range(m):
        for j in range(n):
            result[:,:, i,j] = signal.convolve2d(data[:,:, i,j], mask, boundary='symm', mode='same')
    return result


def generateDataWithDifferentFrequencies_GrayScale(Images, r):
    Images_freq_low = []
    mask = mask_radial(np.zeros([28, 28]), r)
    for i in range(Images.shape[0]):
        fd = fftshift(Images[i, :].reshape([28, 28]))
        fd = fd * mask
        img_low = ifftshift(fd)
        Images_freq_low.append(np.real(img_low).reshape([28 * 28]))

    return np.array(Images_freq_low)

def generateDataWithDifferentFrequencies_3Channel(Images, r):
    Images_freq_low = []
    Images_freq_high = []
    mask = mask_radial(np.zeros([Images.shape[1], Images.shape[2]]), r)
    for i in range(Images.shape[0]):
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * mask
            img_low = ifftshift(fd)
            tmp[:,:,j] = np.real(img_low)
        Images_freq_low.append(tmp)
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * (1 - mask)
            img_high = ifftshift(fd)
            tmp[:,:,j] = np.real(img_high)
        Images_freq_high.append(tmp)

    return np.array(Images_freq_low), np.array(Images_freq_high)
