import cv2
import numpy as np
import glob
import os
import json
from time import time

from mmdet.apis import inference_detector, init_detector


def detect_faces_scrfd(image_path, model, score_thr=0.3):
    # read the image
    img = cv2.imread(image_path)
    # perform inference
    start = time()
    result = inference_detector(model, img)
    duration = time() - start
    # remove all detections with a score lower than the threshold
    result = result[0][result[0][:, 4] > score_thr]
    # count the number of faces
    faces = result.shape[0]
    # draw the bounding boxes
    if len(result) > 0:
        for array in result:
            # get the coordinates of the bounding box
            x1, y1, x2, y2 = array[0], array[1], array[2], array[3]
            # draw the bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    # return the image with the detections
    return img, duration, faces


def main():
    images_path = "./test_images"
    result_path = "./results"
    models_path = "./models"

    models = {
        "2.5g": {"config": "scrfd_2.5g.py", "model": "scrfd_2.5g.pth"},
        "2.5g_kps": {"config": "scrfd_2.5g_bnkps.py", "model": "scrfd_2.5g_bnkps.pth"},
        "10g": {"config": "scrfd_10g.py", "model": "scrfd_10g.pth"},
        "10g_kps": {"config": "scrfd_10g_bnkps.py", "model": "scrfd_10g_bnkps.pth"},
        "34g": {"config": "scrfd_34g.py", "model": "scrfd_34g.pth"},
        "Custom 10g": {"config": "scrfd_10g_custom.py", "model": "custom/latest.pth"},
    }

    device_type = ["cpu", "cuda:0"]

    faces_to_find = [4, 9, 6, 4, 7, 4, 2, 2, 3, 2, 2, 7, 2]

    # read all images in folder
    images = glob.glob(os.path.join(images_path, "*.jpeg"))
    # sort the images by name (taking only the name, and interpreting as an int)
    images = sorted(images, key=lambda x: int(os.path.basename(x).split(".")[0]))
    # create a result folder
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # store the time of detection of each image
    results = {}
    threshold = 0.4

    for device in device_type:
        print(f"Testing device {device}")
        results[device] = {}
        for model_name, model_dict in models.items():
            print(f"Testing model {model_name}")
            # initialize the model
            checkpoint_path = os.path.join(models_path, model_dict["model"])
            config_path = os.path.join("./configs/scrfd", model_dict["config"])
            model = init_detector(config_path, checkpoint_path, device=device)
            # create folder for results
            model_result_path = os.path.join(result_path, model_name)
            # if the folder doesn't exist, create it
            if not os.path.exists(model_result_path):
                os.makedirs(model_result_path)
            # create array for results
            results[device][model_name] = []
            for image in images:
                # get the name of the image as an int
                image_name = int(os.path.basename(image).split(".")[0])
                # perform face detection on the image, using the SCRFD_2.5G_KPS model
                result, duration, faces = detect_faces_scrfd(image, model, score_thr=threshold)
                faces_not_found = faces_to_find[image_name - 1] - faces
                if faces_not_found < 0:
                    faces_not_found = 0
                results[device][model_name].append(
                    {
                        "image": image,
                        "time": duration,
                        "faces found": faces,
                        "faces not found": faces_not_found,
                        "height": result.shape[0],
                        "width": result.shape[1],
                    }
                )
                # save the image with the detections
                cv2.imwrite(os.path.join(model_result_path, os.path.basename(image)), result)

    # output the time of detection of each image
    for device, result in results.items():
        print(f"Device {device}:")
        for model_name, results_list in result.items():
            print(f"Model {model_name}:")
            for element in results_list:
                print(
                    f"Image {element['image'].split(os.path.sep)[-1].split('.')[0]} : {element['time']:.4f} seconds, {element['height']}x{element['width']}"
                )
    # save results as json file
    with open(os.path.join(result_path, "results.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
