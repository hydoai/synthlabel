#from distutils.command.clean import clean
import time
import argparse
import os
import sys
from pathlib import Path
import hashlib
import copy

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]/'yolov5' # yolov5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from bbox_merge import bbox_merge
from iou_tracker import IOUTracker


def random_hash():
    return hashlib.md5(os.urandom(32)).hexdigest()

@torch.no_grad()
def run():
    # PARAMETERS

    # Directories 
    WEIGHTS = ROOT/'weights/yolov5l6.pt'
    SOURCE = 'input_videos'
    PROJECT = 'output' # save results to PROJECT/NAME 
    NAME = 'run' # save results to PROJECT/NAME

    # Inference parameters
    FP16 = True # use fp16 half precision inference
    IMGSZ = (1280,1280)
    CONF_THRES = 0.25
    NMS_THRES = 0.45
    MAX_DET = 100 # NMS iou threshold
    DEVICE = '0' # cuda device number, or 'cpu'
    AGNOSTIC_NMS = False # class-agnostic NMS

    # Object class 
    # Numbers are from 80 class COCO dataset (from 0 to 79)
    CLASSES = [0,1,2,3,5,7] # filter by class (None = all)
    MERGE_CLS = {
        (1,0,'cyclist',80) # (main_box_category_id, sub_box_category_id, merged_category name, merged_category_id)
    } # these new classes will be given category id from 80 and above

    # Saving synthetic labels & images
    SAVE_LABELS = True
    LABEL_CLS = [0,1,2,3,5,7,80] # only generate synthetic image-label sets for these categories
    SKIP_FRAMES = 3 # wait at least this many frames beofre checking if a frame contains objects in 'LABEL_CLS' categories. Recommended < 3, because tracking for retrolabeling requires decent temporal resolution.
    SAVE_ANNOTATED_IMGS = True
    SAVE_CLEAN_IMGS = True

    # Show output
    LINE_THICKNESS = 3 # bounding line thickness
    VIEW_IMG = False # show results on screen



    
    # load models
    device = select_device(DEVICE)
    model = DetectMultiBackend(WEIGHTS, device=device, dnn=False, data=None)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(IMGSZ, s=stride)  # check image size

    # create new categories from merge list
    cls_names = names
    for merge in MERGE_CLS:
        cls_names.append(merge[2])

    # Half
    FP16 &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if FP16 else model.model.float()

    # Dataloader
    dataset = LoadImages(SOURCE, img_size=imgsz, stride=stride, auto=pt)
    bs = 1 # batch_size

    random_video_id = random_hash()

    iou_tracker = IOUTracker()

    # Run inference
    model.warmup(imgsz = (1,3,*imgsz), half=FP16) # run once to warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for frame_index, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        if frame_index % SKIP_FRAMES != 0:
            continue

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if FP16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None] # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1


        # directories
        source_path = Path(path)
        video_filename = source_path.stem
        NAME = video_filename

        save_dir = increment_path(Path(PROJECT) / NAME, exist_ok=True)
        (save_dir/'labels').mkdir(parents =True, exist_ok=True)
        clean_img_save_dir = save_dir/'clean_images'
        (save_dir/'clean_images').mkdir(parents =True, exist_ok=True)
        annotated_img_save_dir = save_dir/'annotated_images'
        (save_dir/'annotated_images').mkdir(parents =True, exist_ok=True)

        # inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        pred = non_max_suppression(pred, conf_thres=CONF_THRES, iou_thres=NMS_THRES, max_det=MAX_DET)
        dt[2] += time_sync() - t3

        # merge "bicycle" and "person" classes into new "cyclist" class
        pred_list = []
        for pred_tensor in pred:
            pred_list.append(bbox_merge(pred_tensor, merge[0], merge[1], merge[3], min_iou=0.1))
        pred = pred_list

        # Process predictions
        for i, det in enumerate(pred): # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            txt_path = str(save_dir/'labels'/random_video_id) + f"_{frame}"
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = copy.deepcopy(im0)
            annotator = Annotator(im0, line_width=LINE_THICKNESS, example=str(cls_names))
            if len(det):
                # rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                det_list = []
                for obj in det:
                    xyxy = obj[:4].cpu().numpy().tolist()
                    conf = obj[4].cpu().numpy()
                    cls = int(obj[5].cpu().numpy())
                    det_list.append({
                        'bbox': xyxy,
                        'score': conf,
                        'class': cls,
                    })

                iou_output = iou_tracker.update(det_list)
                tracked_det = torch.zeros((len(iou_output), 7))
                for i, track in enumerate(iou_output):
                    tracked_det[i][0:4] = torch.tensor(track['bboxes'])
                    tracked_det[i][4] = det[i][4]
                    tracked_det[i][5] = torch.tensor(track['classes'])
                    tracked_det[i][6] = torch.tensor(track['tracking_id'])

                # write results
                # Similar to MOT Challenge format, with no world x,y,z coordinates and additional class_id column
                # https://motchallenge.net/instructions/
                # bounding box is normalized to [0,1]
                # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class_id>
                for *xyxy, conf, cls, track_id in reversed(tracked_det):
                    track_id = int(track_id)
                    if SAVE_LABELS:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1,4)) / gn).view(-1).tolist() # normalized xywh
                        line = (frame, track_id, *xywh, conf, cls)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g,' * len(line)).rstrip() % line + '\n')
                    if SAVE_ANNOTATED_IMGS or VIEW_IMG:
                        c = int(cls)
                        label = f"{cls_names[c]} {track_id}"
                        annotator.box_label(xyxy, label, color=colors(c, True))
                LOGGER.info(f'{s} Done. ({t3-t2:.3f} seconds)')

                # Stream results


                im0 = annotator.result()
                if VIEW_IMG:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)
                
                if SAVE_CLEAN_IMGS:
                    filename = Path(random_video_id + '_' + str(int(frame))+'.jpg')
                    cv2.imwrite(str(clean_img_save_dir/filename), imc)          

                if SAVE_ANNOTATED_IMGS:
                    filename = Path(random_video_id + '_' + str(int(frame))+'.jpg')
                    cv2.imwrite(str(annotated_img_save_dir/filename), im0)          
                








if __name__ == "__main__":
    run()