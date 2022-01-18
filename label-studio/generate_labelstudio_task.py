import numpy as np
from pathlib import Path
import os, hashlib
import json
import rebox
from rebox import BBox
from rebox.formats import yolo, label_studio
import imagesize
import argparse

category_index_to_str = {
    0: 'pedestrian',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    80: 'cyclist',
}    

def random_hash():
    return hashlib.md5(os.urandom(32)).hexdigest()
def create_labelstudio_task(path, port):
    source_dir = Path(path)
    served_url = f'http://localhost:{port}'
    images_dir = source_dir / 'clean_images'
    predictions_dir = source_dir / 'voted_labels'
    total_tasks = []
    for image_path in images_dir.glob('*'):
        # check that file is supported image type
        if image_path.suffix[1:].lower() not in ['bmp', 'gif', 'jpg', 'png', 'svg', 'webp']:
            raise Warning("There is a file that isn't a supported image format. Please check that the images directory contains only images.")

        image_location = served_url + '/' + str(source_dir.name) + '/' + 'clean_images' + f'/{image_path.name}'
        prediction_path = predictions_dir / Path(image_path.stem + '.txt')

        bboxes = []

        with open(prediction_path, 'r') as f:
            for line in f:
                pred = line.split(',')
                pred[-1] = pred[-1][:-1] # remove newline char
                pred = np.array(pred, dtype=float)
                bboxes.append(pred)

        per_image_result = []

        # efficiently get image size
        width, height = imagesize.get(image_path)

        for result in bboxes:
            pred_class = int(result[-1])
            pred_bbox = result[2:6]

            # convert bbox from yolo to label_studio
            yolo_bbox = BBox(pred_bbox, yolo)
            ls_bbox = yolo_bbox.as_format(label_studio).value
            result_json = {
                "original_width": width,
                    "original_height": height,
                    "image_rotation":0,
                    "value": {
                        "x": ls_bbox[0],
                        "y": ls_bbox[1],
                        "width": ls_bbox[2],
                        "height": ls_bbox[3],
                        "rotation": 0,
                        "rectanglelabels": [
                            f"{str(category_index_to_str[pred_class])}"
                        ]
                    },
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels"
            }
            per_image_result.append(result_json)
            
        if args.skip_human:
            per_image_full_json = {
                "id": 1,
                "data" : {
                    "image" : str(image_location)
                },
                "annotations": [{
                    "id": random_hash(),
                    "result" : per_image_result
                }],
            }
            total_tasks.append(per_image_full_json)
        else:
            per_image_full_json = {
                "id": 1,
                "data": {
                    "image": str(image_location)
                },
                "predictions": [{
                    "id": random_hash(),
                    "result": per_image_result
                }],

            }
            total_tasks.append(per_image_full_json)
        
        
    
    return total_tasks
        

def main(args):
    source_dir = Path(args.dir)
    port = args.port
    served_url = f'http://localhost:{port}'
    task_name = args.task_name
    
    images_dir = source_dir / 'stills'
    predictions_dir = source_dir / 'autolabels'
    
    all_tasks = []
    for video_source in source_dir.glob('*'):
        print(f"Including... {video_source}")
        all_tasks += create_labelstudio_task(video_source, port)
    
    with open(source_dir/f'{task_name}.json', 'w') as outfile:
        json.dump(all_tasks, outfile)
        
    num_images = len(all_tasks)
    print(f"Done. Total number of images: {num_images}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description for this program and its arguments")
    parser.add_argument('--dir', default='output', type=str, help='Source directory. See README for correct subdirectory structure. It is identical to output of autolabel.py')
    parser.add_argument('--port', default=8081, type=float, help='This is a float argument')
    parser.add_argument('--task_name', default='labelstudio-task', type=str, help='Name for generated JSON file')
    parser.add_argument('--skip-human', action='store_true', help='For loading predictions as annotations into Label Studio, to skip human annotation.')
    
    args = parser.parse_args()
    main(args)
