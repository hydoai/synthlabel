import glob
import os
import sys
from pathlib import Path
import hashlib
import copy

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]/'yolov5' # yolov5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # relative

# yolov5 dataset imports
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from bbox_merge import bbox_merge
from iou_tracker import IOUTracker

class LoadImagesMoreInfo:
    # modified YOLOv5 standard dataloader to return additional info as tuple instead of just a string
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, s, self.count, self.nf, self.frame, self.frames

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

    def frames_in_vid(self):
        return self.frames



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
    SKIP_FRAMES = 2 # reduce overall input framerate by this factor to speed up inference. Recommended <= 3, because tracker needs high frame rate.
    SKIP_SAVE_FRAMES = 10 # reduce output by this factor to reduce disk space. Modify freely. 
    SAVE_ONLY_IF_FRAME_CONTAINS_CLASS = [80] # only save synthetic labels if this class is present in the frame. None or empty list = all classes. Use to create specialized datasets.
    SAVE_ANNOTATED_IMGS = False # True for debugging
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
    dataset = LoadImagesMoreInfo(SOURCE, img_size=imgsz, stride=stride, auto=pt)
    bs = 1 # batch_size

    random_video_id = random_hash()

    iou_tracker = IOUTracker()

    # Run inference
    model.warmup(imgsz = (1,3,*imgsz), half=FP16) # run once to warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    with tqdm(total=len(dataset)) as pbar_num_files:
        old_num_file = 0
        with tqdm(total=dataset.frames_in_vid()) as pbar_frames:
            for frame_index, (path, im, im0s, vid_cap, s, count_vid, total_vids, count_frame, total_frames) in enumerate(dataset):
                # if moving onto next video
                if count_vid != old_num_file:                 
                    pbar_frames.reset()
                    old_num_file = count_vid
                    pbar_num_files.set_description(f'Video {count_vid+1}/{total_vids}')
                    pbar_num_files.update(1)
                pbar_frames.update(1)

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

                pred = non_max_suppression(pred, conf_thres=CONF_THRES, iou_thres=NMS_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS, max_det=MAX_DET)
                dt[2] += time_sync() - t3


                # merge "bicycle" and "person" classes into new "cyclist" class
                pred_list = []
                for pred_tensor in pred:
                    pred_list.append(bbox_merge(pred_tensor, merge[0], merge[1], merge[3], min_iou=0.1))
                pred = pred_list

                if SAVE_ONLY_IF_FRAME_CONTAINS_CLASS is not None and len(SAVE_ONLY_IF_FRAME_CONTAINS_CLASS) > 0:
                    if set(SAVE_ONLY_IF_FRAME_CONTAINS_CLASS).isdisjoint(set(pred[0][:, -1].cpu().numpy())):
                        tqdm.write(f"skipping: classes in frame: {pred[0][:,-1].cpu().numpy().tolist()}")
                        continue

                # Process predictions
                for i, det in enumerate(pred): # per image
                    seen += 1
                    if seen % SKIP_SAVE_FRAMES != 0:
                        continue
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
                            obj_np = obj.cpu().numpy()
                            xyxy = obj_np[:4].tolist()
                            conf = obj_np[4]
                            cls = int(obj_np[5])
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
                                    line_to_write = (('%g, ' * len(line)).rstrip() % line)
                                    line_to_write = line_to_write[:-1] + '\n'
                                    f.write(line_to_write)
                                    #f.write(('%g,' * len(line)).rstrip() % line + '\n')
                            if SAVE_ANNOTATED_IMGS or VIEW_IMG:
                                c = int(cls)
                                label = f"{cls_names[c]} {track_id}"
                                annotator.box_label(xyxy, label, color=colors(c, True))
                        if len(tracked_det):
                            im0 = annotator.result()
                            if VIEW_IMG:
                                cv2.imshow(str(p), im0)
                                cv2.waitKey(1)
                            
                            if SAVE_CLEAN_IMGS:
                                filename = Path(random_video_id + '_' + str(int(frame))+'.jpg')
                                cv2.imwrite(str(clean_img_save_dir/filename), imc)          
                            tqdm.write("Saving clean image: " + str(clean_img_save_dir/filename))

                            if SAVE_ANNOTATED_IMGS:
                                filename = Path(random_video_id + '_' + str(int(frame))+'.jpg')
                                cv2.imwrite(str(annotated_img_save_dir/filename), im0)          
                            tqdm.write("Saving annotated image: " + str(annotated_img_save_dir/filename))
                        
                        #tqdm.write(f'{s} Done. ({t3-t2:.3f} seconds)')

if __name__ == "__main__":
    run()