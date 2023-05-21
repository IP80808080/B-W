import cv2
import numpy as np

image = cv2.imread('input.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 10, 7, 21)

equalized_image = cv2.equalizeHist(denoised_image)
_, thresholded_image = cv2.threshold(equalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
smoothed_image = cv2.GaussianBlur(thresholded_image, (5, 5), 0)
edges = cv2.Canny(smoothed_image, 50, 150)
composite_image = np.concatenate((image, cv2.cvtColor(denoised_image, cv2.COLOR_GRAY2BGR),
                                  cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR),
                                  cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR),
                                  cv2.cvtColor(smoothed_image, cv2.COLOR_GRAY2BGR),
                                  cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), axis=1)

max_height = 800
if composite_image.shape[0] > max_height:
    ratio = max_height / composite_image.shape[0]
    composite_image = cv2.resize(composite_image, (int(composite_image.shape[1] * ratio), max_height))

cv2.imshow('Enhanced Image', composite_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output.jpg', composite_image)