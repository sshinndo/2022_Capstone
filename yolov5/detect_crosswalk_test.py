# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Run Code
# python detect_crosswalk_test.py --source 0 --weights best_aug3.pt --conf 0.3 --line-thickness 2 --save-txt --save-conf
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2 as cv

#//////////////



#from torchvision.transforms import ToTensor, ToPILImage
#from PIL import Image

#from google.colab.patches import cv_imshow
#///////////

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
flag=0 #flag는 횡단보도 영역 첫 검출

@torch.no_grad()
def f(x, a1, b1):
    return a1 * x + b1

def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        global flag
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        ''''''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        ''''''
        t2 = time_sync()
        dt[0] += t2 - t1



        #자전거 횡단도로 detection
        print("/////////////////////////////")
        print(a, b)

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        #print(pred[0][0])
        #flag는 횡단보도 검출 여부
        #violate는 횡단보도 영역안에 이륜차 탑승 객체가 있을때 1로 변함
        violate = 0

        #만약 횡단보도가 검출 된다면 
        cal_pred = pred
        print(len(cal_pred[0]))
        for i in range(len(cal_pred[0])):
          if(cal_pred[0][i][5] == 2):
            flag = flag +1
            if(flag == 1):
              stand_x_min = cal_pred[0][i][0]
              stand_x_max = cal_pred[0][i][2]
              stand_y_min = cal_pred[0][i][1]
              stand_y_max = cal_pred[0][i][3]

        # 만약 자전거 횡단도로 위에 있다면
        #abno[0], abno[1]



        #횡단보도가 검출됬을때  1:자전거 4:킥보드 7:오토바이
        if flag >= 1:
          for i in range(len(cal_pred[0])):
            if(cal_pred[0][i][5]== 1):

              vio_stand = (cal_pred[0][i][0] + cal_pred[0][i][2])/2
              vio_stand_y = (cal_pred[0][i][1] + cal_pred[0][i][3]) / 2



              # 자전거 도로에서 자전거 달릴시 위법행위 x:
              if vio_stand > int((vio_stand_y - b) / a):
                  violate = 0

              elif(vio_stand<stand_x_max and vio_stand>stand_x_min):
                violate = 1

            if(cal_pred[0][i][5]== 4):

              vio_stand = (cal_pred[0][i][0] + cal_pred[0][i][2])/2
              if(vio_stand<stand_x_max and vio_stand>stand_x_min):
                violate = 1
               
            if(cal_pred[0][i][5]== 7):

              #자전거 도로에서 오토바이 달릴 시 위법행위 o:
              if vio_stand > int((vio_stand_y - b) / a):
                  violate = 1

              vio_stand = (cal_pred[0][i][0] + cal_pred[0][i][2])/2
              if(vio_stand<stand_x_max and vio_stand>stand_x_min):
                violate = 1


        print(violate)




        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            print(det)
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if(flag >= 1):
                print("###########################################################################")
                red= (0, 0, 255)
                print(int(stand_x_min),int(stand_y_min),int(stand_x_max),int(stand_y_max))
                white = (255,255,255)
                font =  cv2.FONT_HERSHEY_PLAIN
                im0 = cv2.putText(im0, "crosswalk", (int(stand_x_min), int(stand_y_min)), font, 2, white, 1, cv2.LINE_8)
                im0 = cv2.putText(im0, "Find CROSSWALK!! Detecting violate!!", (00,30), font, 2, (255,255,255), cv2.LINE_4)
                im0 = cv2.rectangle(im0,(int(stand_x_min),int(stand_y_min)),(int(stand_x_max),int(stand_y_max)),red,3)

                y1 = b
                y2 = 640 * a + b
                im0 = cv.line(im0, (0, int(y1)), (640, int(y2)), (255, 0, 0), 2)


                if violate ==1:
                    red= (0, 0, 255)
                    # 폰트 지정
                    font =  cv2.FONT_HERSHEY_PLAIN
                    # 이미지에 글자 합성하기
                    im0 = cv2.putText(im0, "violate!!!!", (350, 40), font, 2, red, 1, cv2.LINE_AA)

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                  if violate ==1:
                    red= (0, 0, 255)
                    # 폰트 지정
                    font =  cv2.FONT_HERSHEY_PLAIN
                    # 이미지에 글자 합성하기
                    im0 = cv2.putText(im0, "violate!!!!", (350, 40), font, 2, white, 1, cv2.LINE_AA)
                  cv2.imwrite(save_path, im0)

                else:  # 'video' or 'stream'
                    print(flag)
                    if(flag >= 1):
                          print("###########################################################################")
                          red= (0, 0, 255)
                          print(int(stand_x_min),int(stand_y_min),int(stand_x_max),int(stand_y_max))
                          im0 = cv2.rectangle(im0,(int(stand_x_min),int(stand_y_min)),(int(stand_x_max),int(stand_y_max)),red,3)
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]

                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def detect_bike_road():
    global a
    global b
    import numpy as np

    capture = cv.VideoCapture(0)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    while cv.waitKey(33) < 0:
        ret, frame = capture.read()
        cv.imshow("VideoFrame", frame)

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

    import numpy as np
    import matplotlib.pyplot as plt

    #def f(x, a1, b1):
     #   return a1 * x + b1

    def ransac_line_fitting(x, y, r, t):
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

    xno = id2[1]
    yno = id2[0]
    abno, max = ransac_line_fitting(xno, yno, 0.5, 2)

    print(abno[0], abno[1])

    #y1 = f(0, abno[0], abno[1])
    #y2 = f(src.shape[1], abno[0], abno[1])

    a = abno[0]
    b = abno[1]
    #return a, b

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))

    detect_bike_road()
    #a, b = detect_bike_road()
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)