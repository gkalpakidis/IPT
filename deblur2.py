import cv2
import numpy as np

def wiener(image, kernel_size):
    #Generate kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    #dummy = np.copy(image)

    #Pad kernel to image size and center it
    padded_kernel = np.zeros_like(image, dtype=np.float32)
    kh, kw = kernel.shape
    padded_kernel[:kh, :kw] = kernel
    padded_kernel = np.roll(padded_kernel, -kh // 2, axis=0)
    padded_kernel = np.roll(padded_kernel, -kw // 2, axis=1)

    #FFT of image
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    #FFT of kernel
    #padded_kernel = np.zeros_like(image)
    #kh, kw = kernel.shape
    #padded_kernel[:kh, :kw] = kernel
    kernel_dft = cv2.dft(np.float32(padded_kernel), flags=cv2.DFT_COMPLEX_OUTPUT)

    #Wiener filter formula
    h_conj = np.conj(kernel_dft)
    h_square = kernel_dft * h_conj
    denominator = h_square + 0.01 #avoid division by zero
    deblurred = dft_shift * h_conj / denominator

    #Inverse FFT
    deblurred = np.fft.ifftshift(deblurred)
    deblurred = cv2.idft(deblurred, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    #return np.uint8(deblurred)

    #Normalize result to 0-255 and convert to uint8
    deblurred = cv2.normalize(deblurred, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(deblurred)

def deblur(img_path, output_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    dblr = wiener(img, kernel_size=5)
    cv2.imwrite(output_path, dblr)

deblur("/blurred.png", "/Imaging/deblurred_wiener.png")