import cv2 as cv
import numpy as np
import os

def ransac_line_fitting(x, y, r, t):
    try:
        iter = np.round(np.log(1 - 0.999) / np.log(1 - (1 - r) ** 2) + 1)
        num_max = 0
        for i in np.arange(iter):
            id = np.random.permutation(len(x))
            xs = x[id[:2]]
            ys = y[id[:2]]
            A = np.vstack([xs, np.ones(len(xs))]).T
            ab = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, ys))
            dist = np.abs(ab[0] * x - y + ab[1]) / np.sqrt(ab[0] ** 2 + 1)
            numInliers = sum(dist < t)
            if numInliers > num_max:
                ab_max = ab
                num_max = numInliers
        return ab_max, num_max

    except:
        print("[info] ransec 에러 : 예외처리 되었습니다. 재시도")
        terminal_command = "python detect_crosswalk_test.py --source 0 --weights best_aug3.pt --conf 0.3 --line-thickness 2 --save-txt --save-conf"
        os.system(terminal_command)

def main():
    capture = cv.VideoCapture(0)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    print("[info] Detecting_Bike function is now ONLINE")
    ret, frame = capture.read()

    capture.release()
    cv.destroyAllWindows()

    src = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dst = cv.inRange(src, (170, 30, 0), (180, 255, 255))
    dst2 = cv.inRange(src, (0, 30, 0), (10, 255, 255))

    h4 = cv.addWeighted(dst, 1.0, dst2, 1, 0)

    bike_road = cv.bitwise_and(src, src, mask=h4)
    bike_road = cv.cvtColor(bike_road, cv.COLOR_HSV2BGR)

    bike_road_gray = cv.cvtColor(bike_road, cv.COLOR_BGR2GRAY)

    for i in range(bike_road_gray.shape[1]):
        for j in range(bike_road_gray.shape[0]):
            if (bike_road_gray[j][i] > 0):
                bike_road_gray[j][i] = 255

    bike_road_gray_no = cv.medianBlur(bike_road_gray, 5)

    lx = cv.Sobel(bike_road_gray_no, ddepth=cv.CV_64F, dx=1, dy=0, ksize=3)
    ly = cv.Sobel(bike_road_gray_no, ddepth=cv.CV_64F, dx=0, dy=1, ksize=3)
    mag = np.sqrt(np.square(lx) + np.square(ly))
    ori = np.arctan2(ly, lx) * 180 / np.pi

    lx_ = (lx - lx.min()) / (lx.max() - lx.min()) * 255
    ly_ = (ly - ly.min()) / (ly.max() - ly.min()) * 255
    mag_ = (mag - mag.min()) / (mag.max() - mag.min()) * 255
    ori_ = (ori - ori.min()) / (ori.max() - ori.min()) * 255

    result1 = np.zeros(bike_road_gray_no.shape)
    id1 = np.where(mag > 400)
    result1[id1] = 255

    result2 = np.zeros(bike_road_gray_no.shape)
    id2 = np.where((mag > 100) & (ori > 0) & (ori < 40))
    result2[id2] = 255

    result3 = np.zeros(bike_road_gray_no.shape)
    id3 = np.where((mag > 100) & (ori > -70) & (ori < 0))
    result3[id3] = 255

    xno = id2[1]
    yno = id2[0]
    abno, max = ransac_line_fitting(xno, yno, 0.5, 2)

    print(abno[0], abno[1])

    # y1 = f(0, abno[0], abno[1])
    # y2 = f(src.shape[1], abno[0], abno[1])

    return abno[0], abno[1]

if __name__ == '__main__':
    main()