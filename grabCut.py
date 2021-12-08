import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('Soldier.png')
mask = np.zeros(img.shape[:2],np.uint8)

iterCount = 5

rect = (50,50,450,290)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

model = cv.GC_INIT_WITH_RECT

cv.grabCut(img,mask,rect,bgdModel,fgdModel,iterCount,model)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img),plt.colorbar(),plt.show()