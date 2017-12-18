import cv2 as cv
import numpy as np
import libtiff as tiff
import pylab as pl

def build_filters():
	filters = []
	ksize = [7,8,11,13,15,17] ## gabor size
	lamda = np.pi/2.0 ## bo chang
	for theta in np.arange(0,np.pi, np.pi/4):
		for k in range(6):
			kern = cv.getGaborKernel((ksize[k],ksize[k]),1.0, theta,lamda,0.5,0,ktype=cv.CV_32F)
			kern /= 1.5*kern.sum()
			filters.append(kern)
	return filters

def process(img, filters):
	accum = np.zeros_like(img)
	for kern in filters:
		fimg = cv.filter2D(img,cv.CV_8UC3, kern)
		np.maximum(accum, fimg, accum)
	return accum

def getGabor(img, filters):
	res = [] ## lv bo jie jie guo (the result after filter)
	for i in range(len(filters)):
		res1 = process(img, filters[i])
		res.append(np.array(res1))

	pl.figure(2)
	for temp in range(len(res)):
		pl.subplot(4,6,temp+1)
		pl.imshow(res[temp], cmap='gray')
	pl.show()


img = cv.imread("/home/ai/data/jaffe//KA.AN1.39.tiff", 0)
print(img.shape)
a=build_filters()
print(len(a))
accum = process(img, a)
getGabor(img, a)
