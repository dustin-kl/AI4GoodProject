import cv2 as cv
import os
import os.path as osp


i_path = './dataset/kaggle/training/groundtruth_ori/'
o_path = './dataset/kaggle/training/groundtruth/'


def main():
    files = [f for f in os.listdir(i_path) if f.endswith('png')]
    files.sort()
    os.makedirs(o_path, exist_ok=True)

    for f in files:
        img = cv.imread(osp.join(i_path, f), cv.IMREAD_GRAYSCALE)
        img[img == 255] = 1
        cv.imwrite(osp.join(o_path, f), img, )


if __name__ == '__main__':
    main()