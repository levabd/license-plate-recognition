import os
import cPickle
import tools as tl
import cv2

from thresholding import threshold_niblack
from skimage import img_as_ubyte

import numpy as np
import debugtools as dtl

# Delete all files
folders = ["Segmentation"]
for folder in folders:
    for the_file in os.listdir("debug_imgs/"+folder):
        file_path = os.path.join("debug_imgs/"+folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

ranges = cPickle.load(open( "Chars.p", "r" ))

# setup some parameters
scale = 0.5

# minPlateW = scale * 60
# maxPlateW = scale * 180

# minPlateH = minPlateW/4.64
# maxPlateH = maxPlateW/4.64

framenum = 0
categories = ["Dirty LPs", "LPs big"]
totalChars = np.zeros(2, dtype=np.int)
foundedChars = np.zeros(2, dtype=np.int)
totalPlates = np.zeros(2, dtype=np.int)
fullPlates = np.zeros(2, dtype=np.int)


def AddSegment(xm, color):
    xl = tl.findb0(horp, xm, 0.7 * horp[xm])
    xr = tl.findb1(horp, xm, 0.7 * horp[xm])
    horp[xl:xr] = 0

    if len(verticalSegments) > 0:
        if np.isclose(xm, verticalSegments, atol=15).any():
            return

    verticalSegments.append(xm)
    cv2.line(imgBand, (xm, 0), (xm, imgBand.shape[0]), color, 1)
    cv2.line(bright, (xm, 0), (xm, bright.shape[0]), (0, 0, 255), 1)
    cv2.line(equ, (xm, 0), (xm, equ.shape[0]), (0, 0, 255), 1)
    cv2.line(thres, (xm, 0), (xm, thres.shape[0]), (0, 0, 255), 1)
    cv2.line(_horpDr, (xm, 0), (xm, _horpDr.shape[0]), (0, 0, 255), 1)


for (imagepath, lpranges) in ranges:
    print imagepath + ' ' +str(framenum)
    # print len(lpranges)

    img = cv2.imread(imagepath)

    # get mask
    maskpath = imagepath.replace('.jpg','M.jpg')
    mask = cv2.imread(maskpath)
    mask = cv2.resize(mask,(0,0),fx=scale,fy=scale)

    # get char mask
    maskCharpath = imagepath.replace('.jpg','C.jpg')
    maskChar = cv2.imread(maskCharpath)

    '''''''''''''''''''''''''''''''''''''''
    Get the needed range according to mask
    '''''''''''''''''''''''''''''''''''''''

    bestRange = []
    maxScore = 0.0
    maxPer = 0.0

    for r in lpranges:
        rsum = np.count_nonzero(mask[r[0]:r[1],r[2]:r[3]])
        per = rsum/((r[1]-r[0])*(r[3]-r[2]))

        if rsum * per > maxScore * maxPer:
            maxScore = rsum
            maxPer = per
            bestRange = r

    yl = int(bestRange[0]/scale)
    yr = int(bestRange[1]/scale)
    xl = int(bestRange[2]/scale)
    xr = int(bestRange[3]/scale)

    imgBand = img[yl:yr,xl:xr].copy()
    maskCharBand = maskChar[yl:yr,xl:xr].copy()

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Preprocessing
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    # To gray
    gray = cv2.cvtColor(imgBand, cv2.COLOR_BGR2GRAY)

    # Sharpness
    # kernel = np.matrix([
    #    [-0.1, -0.1, -0.1],
    #    [-0.1,    2, -0.1],
    #    [-0.1, -0.1, -0.1]
    # ])

    # sharp = cv2.filter2D(equ, -1, kernel)

    # Brightness
    kernel = np.matrix([
        [-0.1, 0.1, -0.1],
        [ 0.1,   2,  0.1],
        [-0.1, 0.1, -0.1]
    ])

    bright = cv2.filter2D(gray, -1, kernel)

    # Histogram equalization
    equ = cv2.equalizeHist(bright)
    # equ = gray

    # Otsu binarization
    # ret, thres = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY +
    #                            cv2.THRESH_OTSU)

    # Simple adaptive binarization
    # thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                               cv2.THRESH_BINARY, 11, 3)

    # Niblack binarization
    thresh_niblack = threshold_niblack(equ, window_size=25, k=0.3)
    binary_niblack = equ > thresh_niblack
    thres = img_as_ubyte(binary_niblack)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Plate segmentation
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    horp = tl.projectionHor(thres)
    _horpDr = tl.getDrawProjectionHor(thres, horp)
    vm = np.max(horp)
    va = np.mean(horp)
    vb = 2*va-vm

    verticalSegments = []

    while True:
        xm = np.argmax(horp)
        if horp[xm] > 0.87 * vm:
            AddSegment(xm, (0, 0, 255))
        else:
            break

    lh0 = thres.shape[1] * 45 / 100
    rh0 = thres.shape[1] * 50 / 100

    while True:
        xm = np.argmax(horp[lh0:rh0]) + lh0
        if horp[xm] > 0.84 * vm:
            AddSegment(xm, (255, 0, 0))
        else:
            break

    lh1 = thres.shape[1] * 55 / 100
    rh1 = thres.shape[1] * 60 / 100

    while True:
        xm = np.argmax(horp[lh1:rh1]) + lh1
        if horp[xm] > 0.82 * vm:
            AddSegment(xm, (255, 0, 0))
        else:
            break

    lh2 = thres.shape[1] * 65 / 100
    rh2 = thres.shape[1] * 70 / 100

    while True:
        xm = np.argmax(horp[lh2:rh2]) + lh2
        if horp[xm] > 0.84 * vm:
            AddSegment(xm, (255, 0, 0))
        else:
            break

    verticalSegments.append(0)
    verticalSegments.append(imgBand.shape[1]-1)
    verticalSegments = np.asarray(verticalSegments, dtype=np.int)
    verticalSegments.sort()
    # print verticalSegments

    total, founded = dtl.getSegmentation1Accuracy(verticalSegments, maskCharBand)

    indx = 1 if imagepath.find(categories[0]) != -1 else 0
    totalPlates[indx] += 1
    totalChars[indx] += total
    foundedChars[indx] += founded
    if total == founded:
        fullPlates[indx] += 1

    framenum += 1

    fimg = tl.concat_ver((imgBand, bright, equ, thres, _horpDr))

    cv2.imwrite("debug_imgs/Segmentation/"+str(framenum)+".jpg", fimg)

for i in range(len(categories)):
    print ""
    print "##########", categories[i - 1]
    print "########## From %d characters founded %d not founded %d" % (totalChars[i], foundedChars[i], totalChars[i]-foundedChars[i])
    print "########## From %d plates totally segmented %d not segmented %d" % (totalPlates[i], fullPlates[i], totalPlates[i]-fullPlates[i])