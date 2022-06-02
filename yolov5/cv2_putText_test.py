import numpy as np
import cv2


bg_img = np.zeros((480, 320, 3), np.uint8)
bg_img = cv2.putText(bg_img, "USER GUI", (90, 30), 3, 0.8, (255, 255, 255), 0)
bg_img = cv2.putText(bg_img, "Crosswalk Set!", (40, 80), 0, 1, (0, 0, 255), 2)
bg_img = cv2.putText(bg_img, "Detected Violate", (30, 130), 0, 1, (0, 0, 255), 3)

while(1):
    cv2.imshow("test", bg_img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break