import os
import cPickle
import tools as tl
import cv2
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
    gray = cv2.cvtColor(imgBand, cv2.COLOR_BGR2GRAY)


    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Plate segmentation
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
    thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)

    horp = tl.projectionHor(thres)
    _horpDr = tl.getDrawProjectionHor(gray, horp)
    vm = np.max(horp)
    va = np.mean(horp)
    vb = 2*va-vm

    verticalSegments = []
    while True:
        xm = np.argmax(horp)
        if horp[xm] > 0.86 * vm: 
            xl = tl.findb0(horp, xm, 0.7 * horp[xm])
            xr = tl.findb1(horp, xm, 0.7 * horp[xm])
            horp[xl:xr] = 0
            verticalSegments.append(xm)
            cv2.line(imgBand,(xm,0),(xm,imgBand.shape[0]),(0,0,255),1)
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

    fimg = tl.concat_ver((imgBand, thres, _horpDr))
    #cv2.imshow("Result", cv2.resize(fimg,(0,0),fx=3,fy=3))
    #cv2.waitKey(0)

    cv2.imwrite("debug_imgs/Segmentation/"+str(framenum)+".jpg", fimg)

for i in range(len(categories)):
    print ""
    print "##########", categories[i - 1]
    print "########## From %d characters founded %d not founded %d" % (totalChars[i], foundedChars[i], totalChars[i]-foundedChars[i])
    print "########## From %d plates totally segmented %d not segmented %d" % (totalPlates[i], fullPlates[i], totalPlates[i]-fullPlates[i])