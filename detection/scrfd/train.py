import os
import random
import shutil
import copy
import os.path as osp
import time
import glob

import mmcv
from mmdet.datasets.retinaface import RetinaFaceDataset
import torch
import cv2
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash


from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.apis import inference_detector, init_detector

from torch.utils.data import DataLoader


def detect_faces_scrfd(image_path, model, score_thr=0.3):
    # read the image
    img = cv2.imread(image_path)
    # perform inference
    result = inference_detector(model, img)
    # remove all detections with a score lower than the threshold
    result = result[0][result[0][:, 4] > score_thr]
    # count the number of faces
    faces = result.shape[0]
    # return the image with the detections and the bounding boxes
    return img, result, faces


train_path = "data/retinaface/train"
train_images_path = "data/retinaface/train/images"
train_annotation_path = "data/retinaface/train/labelv2.txt"
val_path = "data/retinaface/val"
val_images_path = "data/retinaface/val/images"
val_annotation_path = "data/retinaface/val/labelv2.txt"
test_path = "data/retinaface/test"
test_images_path = "data/retinaface/test/images"
test_annotation_path = "data/retinaface/test/labelv2.txt"

models = {
    "2.5g": {"config": "scrfd_2.5g.py", "model": "scrfd_2.5g.pth"},
    "2.5g_kps": {"config": "scrfd_2.5g_bnkps.py", "model": "scrfd_2.5g_bnkps.pth"},
    "10g": {"config": "scrfd_10g.py", "model": "scrfd_10g.pth"},
    "10g_kps": {"config": "scrfd_10g_bnkps.py", "model": "scrfd_10g_bnkps.pth"},
    "34g": {"config": "scrfd_34g.py", "model": "scrfd_34g.pth"},
}
models_path = "./models"


def dataset_prep(dataset_path: str, selected_model: dict):
    print("Preparing dataset...")
    start_image_shuffle = time.time()
    print("Shuffling images and splitting them into train, val and test...")
    # grab all jpg from dataset_path
    # images are in subfolders, so we need to grab them recursively
    files = []
    for r, d, f in os.walk(dataset_path):
        for file in f:
            if ".jpg" in file:
                files.append(os.path.join(r.split("/")[-1], file))

    # now mix them up random and split them into train, val, test with 80/10/10 split
    random.shuffle(files)
    train = files[: int(len(files) * 0.8)]
    val = files[int(len(files) * 0.8) : int(len(files) * 0.9)]
    test = files[int(len(files) * 0.9) :]

    # if the folders don't exist, create them (recursively, since the folder is /data/train/images)
    if not os.path.exists(train_images_path):
        os.makedirs(train_images_path)
    if not os.path.exists(val_images_path):
        os.makedirs(val_images_path)
    if not os.path.exists(test_images_path):
        os.makedirs(test_images_path)

    # now copy them to their respective folders
    for f in train:
        # extract the folder in front of the filename, if it doesn't exist, create it
        folder = f.split("/")[0]
        if not os.path.exists(os.path.join(train_images_path, folder)):
            os.makedirs(os.path.join(train_images_path, folder))
        shutil.copy2(os.path.join(dataset_path, f), os.path.join(train_images_path, f))
    for f in val:
        folder = f.split("/")[0]
        if not os.path.exists(os.path.join(val_images_path, folder)):
            os.makedirs(os.path.join(val_images_path, folder))
        shutil.copy2(os.path.join(dataset_path, f), os.path.join(val_images_path, f))
    for f in test:
        folder = f.split("/")[0]
        if not os.path.exists(os.path.join(test_images_path, folder)):
            os.makedirs(os.path.join(test_images_path, folder))
        shutil.copy2(os.path.join(dataset_path, f), os.path.join(test_images_path, f))

    end_image_shuffle = time.time()
    print(f"Done in {end_image_shuffle - start_image_shuffle:.2f}s")

    print("Detecting faces and writing annotations...")
    start_annotation = time.time()

    annotations = [
        (train_annotation_path, train_images_path),
        (val_annotation_path, val_images_path),
        (test_annotation_path, test_images_path),
    ]

    # load the model

    checkpoint_path = os.path.join(models_path, selected_model["model"])
    config_path = os.path.join("./configs/scrfd", selected_model["config"])

    model = init_detector(config_path, checkpoint_path, device="cuda:0")

    for annotation_path, images_path in annotations:
        print(f"Writing annotation file {annotation_path}...")
        start_current_annotation = time.time()
        # open the annotation file
        with open(annotation_path, "w") as f:
            # for every image in the folder, detect the faces and write the annotation. Images may be in subfolders
            for root, dirs, files in os.walk(images_path):
                for image in files:
                    # detect the faces
                    image_path = os.path.join(root, image)
                    img, result, faces = detect_faces_scrfd(image_path, model, score_thr=0.4)
                    # the annotation is of the form :
                    # <image_path> image_width image_height
                    # bbox_x1 bbox_y1 bbox_x2 bbox_y2 (<keypoint,3>*N)
                    # ...
                    # ...

                    # write the image path from root, width and height
                    name = root.split("/")[-1] + "/" + image
                    f.write(f"# {name} {img.shape[1]} {img.shape[0]}\n")
                    # write the bounding boxes and keypoints
                    for bbox in result:
                        f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[0]}\n")
        print(f"Done in {time.time() - start_current_annotation:.2f}s")

    end_annotation = time.time()
    print(f"Done in {end_annotation - start_annotation:.2f}s")


def train_model(config_file: str, model_save_dir: str):
    print("Training model...")
    start_training = time.time()
    cfg = Config.fromfile(config_file)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.work_dir = model_save_dir
    cfg.gpu_ids = range(1)

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(config_file)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    # logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text
    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    cfg.seed = 0
    meta["seed"] = 0
    meta["exp_name"] = osp.basename(config_file)

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets: list[RetinaFaceDataset] = [build_dataset(cfg.data.train)]

    # print information about the dataset
    logger.info(f"Loaded {len(datasets)} datasets")
    for dataset in datasets:
        print(dataset.data_root)

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        timestamp=timestamp,
        meta=meta,
    )
    end_training = time.time()
    print(f"Training done in {end_training - start_training:.2f}s")

    print("Training done, now testing...")
    start_test = time.time()
    # once the model is trained, verify the results with the test dataset
    images = glob.glob(os.path.join(test_images_path, "**/*.jpg"), recursive=True)

    annotations = {}
    faces_to_find = 0
    # open test annotation file
    with open(test_annotation_path, "r") as f:
        # first line starts with a #, contains name, width and height seperated by spaces
        # then next lines contain the bounding boxes and keypoints of the faces in the image
        # then again a line starting with # for the next image , and so on
        current_image = ""
        for line in f:
            # if the line is an image, create a dictionary entry with the name and a list of bounding boxes
            if line.startswith("#"):
                line = line.split()
                current_image = line[1].split("/")[-1]
                annotations[current_image] = []
            # if the line is a bounding box, add it to the list of bounding boxes
            else:
                line = line.split()
                annotations[current_image].append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
                faces_to_find += 1

    # load the model
    model = {"config": "scrfd_10g_custom.py", "model": "custom/latest.pth"}
    checkpoint_path = os.path.join(models_path, model["model"])
    config_path = os.path.join("./configs/scrfd", model["config"])
    model = init_detector(config_path, checkpoint_path, device="cuda:0")
    # run the model on the test images
    faces_found = 0
    correct_faces = 0
    for image in images:
        # get the image name
        name = image.split("/")[-1]
        # load the image
        img = cv2.imread(image)
        # run the model on the image with the detect_faces_scrfd
        img, result, faces = detect_faces_scrfd(image, model, score_thr=0.4)
        # if the image is in the annotations, check if the bounding boxes are correct
        if name in annotations:
            # for each bounding box in the annotations
            for annotation in annotations[name]:
                # for each bounding box in the result
                for box in result:
                    # if the bounding box is close enough to the annotation, add one to the faces_found
                    faces_found += 1
                    if (
                        abs(box[0] - annotation[0]) < 60
                        and abs(box[1] - annotation[1]) < 60
                        and abs(box[2] - annotation[2]) < 60
                        and abs(box[3] - annotation[3]) < 60
                    ):
                        correct_faces += 1
                        break

    test_time = time.time() - start_test
    print(f"Testing done in {test_time:.2f}s")
    # print the results
    print(
        f"For {len(images)} images, {faces_found} faces were found, {correct_faces} of them were correct, {faces_to_find} faces were annotated"
    )


if __name__ == "__main__":
    # if data folder doesn't exist, run the function
    if not os.path.exists("./data"):
        dataset_prep("./dataset", models["10g"])

    # train the model based on the 10g model
    train_model("./configs/scrfd/scrfd_10g_custom.py", "./models/custom")
