import cv2
import numpy as np

a = cv2.imread("Data/test_scene/acnewash.png")
a[a==255] = 1
print(a, np.max(a), a.dtype, np.unique(a), a.shape, a[:,:,0])