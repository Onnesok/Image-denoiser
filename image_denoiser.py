import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt



def wavelet_denoise(image, wavelet='haar', level=2):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(gray_image, wavelet, level=level)
    
    threshold = np.sqrt(2 * np.log(gray_image.size)) * 0.1
    coeffs_thresh = list(coeffs)
    coeffs_thresh[1:] = [tuple(pywt.threshold(c, threshold, mode='soft') for c in detail) for detail in coeffs[1:]]
    
    # Reconstruct the image
    denoised_image = pywt.waverec2(coeffs_thresh, wavelet)
    
    # Clip values to the valid range and convert to uint8 and done
    denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)
    
    return denoised_image



def denoise_image(image_path):
    noisy_image = cv2.imread(image_path)
    noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
    
    
    gaussian_denoised_image = cv2.GaussianBlur(noisy_image, (7,7), sigmaX=1.5)
    gaussian_denoised_image_rgb = cv2.cvtColor(gaussian_denoised_image, cv2.COLOR_BGR2RGB)
    
    median_denoised_image = cv2.medianBlur(noisy_image, ksize=5)
    median_denoised_image_rgb = cv2.cvtColor(median_denoised_image, cv2.COLOR_BGR2RGB)
    
    bilateral_denoised_image = cv2.bilateralFilter(noisy_image, d=9, sigmaColor=75, sigmaSpace=75)
    bilateral_denoised_image_rgb = cv2.cvtColor(bilateral_denoised_image, cv2.COLOR_BGR2RGB)
    

    nl_means_denoised_image = cv2.fastNlMeansDenoisingColored(noisy_image, None, 10, 10, 7, 21)
    nl_means_denoised_image_rgb = cv2.cvtColor(nl_means_denoised_image, cv2.COLOR_BGR2RGB)
    
    wavelet_denoised_image = wavelet_denoise(noisy_image)
    wavelet_denoised_image_rgb = cv2.cvtColor(wavelet_denoised_image, cv2.COLOR_GRAY2RGB)
    
    plt.figure(figsize=(10, 10))
    
    
    plt.subplot(3, 3, 1)
    plt.title("Noisy Image")
    plt.imshow(noisy_image_rgb)
    plt.axis("off")
    
    
    #plt.figure(figsize=(15, 10))
    plt.subplot(3, 3, 2)
    plt.title("Gaussian Blur Image")
    plt.imshow(gaussian_denoised_image_rgb)
    plt.axis("off")
    

    plt.subplot(3, 3, 3)
    plt.title("Median blur image")
    plt.imshow(median_denoised_image_rgb)
    plt.axis("off")
    
    
    plt.subplot(3, 3, 4)
    plt.title("Bilateral filter image")
    plt.imshow(bilateral_denoised_image)
    plt.axis("off")
    
    
    plt.subplot(3, 3, 5)
    plt.title("Non-local denoised image")
    plt.imshow(nl_means_denoised_image_rgb)
    plt.axis("off")
    
    plt.subplot(3, 3, 6)
    plt.title("wavelet denoised image")
    plt.imshow(wavelet_denoised_image)
    plt.axis("off")
    

    plt.show()
    
    
    
    
denoise_image("noisy.png")
denoise_image("noisy1.png")
    