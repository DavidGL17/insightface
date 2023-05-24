import os

# set the environment variables
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
os.environ["MXNET_CUDNN_LIB_CHECKING"] = "0"
import cv2
import sys
import numpy as np
import glob
from retinaface import RetinaFace
from time import time
import json


# if flag --results is passed, create a boolean variable to store it
if "--results" in sys.argv:
    results = True
else:
    results = False

thresh = 0.8


count = 1

gpuid = 0
detector = RetinaFace("./model/R50", 0, gpuid, "net3")

images_path = "./test_images"
result_path = "./results"
# read all images in folder
images = glob.glob(os.path.join(images_path, "*.jpeg"))
images = sorted(images, key=lambda x: int(os.path.basename(x).split(".")[0]))
# create a result folder
if not os.path.exists(result_path):
    os.makedirs(result_path)
# store the time of detection of each image
time_list = []
if results:
    # if flag --results is passed, read the results.json file
    with open(os.path.join(result_path, "results.json"), "r") as f:
        time_list = json.load(f)

if not results:
    # first do a warmup with any image
    scales = [1024, 1980]
    img = cv2.imread(images[0])
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # im_scale = 1.0
    # if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [im_scale]
    flip = False
    detector.detect(img, thresh, scales=scales, do_flip=flip)

    for image in images:
        scales = [1024, 1980]
        print(f"Processing {image}...")
        img = cv2.imread(image)
        print(img.shape)
        im_shape = img.shape
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        # im_scale = 1.0
        # if im_size_min>target_size or im_size_max>max_size:
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        print("im_scale", im_scale)

        scales = [im_scale]
        flip = False

        for c in range(count):
            start = time()
            faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
            result = time() - start
            print(c, faces.shape, landmarks.shape)

        if faces is not None:
            # store name, time, number of faces and height and width of image
            time_list.append(
                {
                    "image": image,
                    "time": result,
                    "error rate": 0,
                    "height": img.shape[0],
                    "width": img.shape[1],
                }
            )
            print("find", faces.shape[0], "faces")
            for i in range(faces.shape[0]):
                # print('score', faces[i][4])
                box = faces[i].astype(np.int)
                # color = (255,0,0)
                color = (0, 0, 255)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                if landmarks is not None:
                    landmark5 = landmarks[i].astype(np.int)
                    # print(landmark.shape)
                    for l in range(landmark5.shape[0]):
                        color = (0, 0, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
            name = image.split(os.path.sep)[-1].split(".")[0]
            filename = os.path.join(result_path, f"{name}_result.jpg")
            print("writing", filename)
            cv2.imwrite(filename, img)
        else:
            time_list.append(
                {
                    "image": image,
                    "time": result,
                    "error rate": 0,
                    "height": img.shape[0],
                    "width": img.shape[1],
                }
            )
# save results as json file
with open(os.path.join(result_path, "results.json"), "w") as f:
    json.dump(time_list, f)
# for the first element of each tuple, remove the path and the extension, and sort it as a number
time_list.sort(key=lambda x: int(x["image"].split(os.path.sep)[-1].split(".")[0]))
# output the time of detection of each image
for element in time_list:
    print(
        f"Image {element['image'].split(os.path.sep)[-1].split('.')[0]} : {element['time']} seconds, {element['error rate']} error rate, {element['height']}x{element['width']}"
    )
