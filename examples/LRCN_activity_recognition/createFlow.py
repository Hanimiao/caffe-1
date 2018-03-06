from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import pyflow
import os

from concurrent.futures import ProcessPoolExecutor, as_completed

alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0

def createFlow(base, savebase, impath1, impath2):
    im1 = np.array(Image.open(os.path.join(base, impath1)))
    im2 = np.array(Image.open(os.path.join(base, impath2)))

    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    #s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    #e = time.time()

    #print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
    #    e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)

    import cv2
    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(savebase, impath2), rgb)


def create(base, savebase):
    futures = set()
    with ProcessPoolExecutor() as executor:

        frames = os.listdir(os.path.join(base))
        frames.sort()
        #print(frames)
        for i in xrange(11, 27, 1):
            future = executor.submit(createFlow, base, savebase, frames[i], frames[i+1])
            futures.add(future)
    try:
        for future in as_completed(futures):
            err = future.exception()
            if err is not None:
                raise err
    except KeyboardInterrupt:
        print("stopped by hand")



# import time
# s = time.time()
# create('/home/link/data/Emotion/web/temp/link/rgbTemp', '/home/link/data/Emotion/web/temp/link/flowTemp')
# e = time.time()
# print(e-s)

