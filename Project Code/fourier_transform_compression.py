import cv2
import numpy as np

# Read image and transcribe the RGB values into a matrix
M = cv2.imread("Images/dog-image.jpg")

# Take Fast Fourier Transform (FFT)
FM = np.fft.fft2(M)

# Discard all values with magnitude less than the tolerance (compression)
NFM = FM

# Start by sorting the values by magnitude
NFM_Sorted = np.sort(np.abs(NFM.reshape(-1)))

# Percentage of largest data to keep (K)
K = 0.004

# Find threshold value based on percentage
threshold = NFM_Sorted[int(round((1-K) * NFM_Sorted.size))]

# Zero out all values below the threshold
NFM[np.abs(NFM) < threshold] = 0

# Take the Inverse Fast Fourier Transform (IFFT) and take the real part to discard 
# of the tiny complex bits (on the order of 1e-15)
NM = np.real(np.fft.ifft2(NFM))

# Write compressed data to image
cv2.imwrite('Images/supercompressed-image.png', NM)