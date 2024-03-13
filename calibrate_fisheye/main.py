import numpy as np
import cv2
import os
import glob

#second
DIM=(748, 748)
K=np.array([[198.8740791638536, 0.0, 372.60696548881566], [0.0, 199.164517591797, 376.11482369572565], [0.0, 0.0, 1.0]])
D=np.array([[0.11051804040817755], [-0.0739384081419349], [0.05755843084810656], [-0.019397708044220068]])

def undistort(img_path):
    img = cv2.imread(img_path, 0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imwrite('output/first/out_' + os.path.basename(img_path), undistorted_img)

if __name__ == '__main__':
    images = glob.glob('source/resized/*.jpg')
    for p in images:
        undistort(p)