import cv2
import numpy as np
import pywt

# Read image and transcribe the RGB values into a matrix
M = cv2.imread("Images/dog-image.jpg")

colors = []
for color in cv2.split(M):

    # Take Discrete Wavelet Transform (DWT)
    coefficients = pywt.wavedec2(color, 'haar', level=4)
    coeffs, coeffs_slices = pywt.coeffs_to_array(coefficients)


    # Discard all values with magnitude less than the tolerance (compression)
    coeffs_w = coeffs
    # Start by sorting the values by magnitude
    coeffs_sorted = np.sort(np.abs(coeffs.reshape(-1)))

    # Percentage of largest data to keep
    percent = 0.004
    # Find threshold value based on percentage
    threshold = coeffs_sorted[int(round((1-percent) * coeffs_sorted.size))]
    # Zero out all values below the threshold
    coeffs_w[np.abs(coeffs_w) < threshold] = 0

    # Take the Inverse Discrete Wavelet Transform (IDWT) 
    coeffs_comp = pywt.array_to_coeffs(coeffs_w, coeffs_slices, 'wavedec2')
    img_comp = pywt.waverec2(coeffs_comp, 'haar')

    colors.append(img_comp)

img = cv2.merge(colors)

# Write compressed data to image
cv2.imwrite('Images/wavelet-supercompressed-image.png', img)

