import json
import os
from os.path import basename
from pathlib import Path
from re import I
import imagesize
from rebox import BBox
from rebox.formats import yolo, coco
import shutil
import argparse



#new_image_path = os.path.join(Path.cwd().parent / 'finished', 'images')

# create new 'finished/images' folder if it doesn't exist and delete it if it does
def run(args):
    created_json = {}

    created_json.update(
        {'categories': [
            {'id': 0, 'name': 'bicycle'},
            {'id': 1, 'name': 'bus'},
            {'id': 2, 'name': 'car'},
            {'id': 3, 'name': 'cyclist'},
            #{'id': 4, 'name': 'motorcycle'},
            {'id': 4, 'name': 'pedestrian'},
            {'id': 5, 'name': 'truck'}
        ]}
    )

    created_json.update(
        {'info': {
            'year':2021,
            'version':'1.0',}
        }
    )

    autolabel_to_alphabetical_categories = {
        0 : 4, # person -> pedestrian
        1 : 0, # bicycle -> bicycle
        2 : 2, # car -> car
        #3 : 4, # motorcycle # temporarily skipped because this time i didn't detect motorcycles in autolabel.py
        5 : 1, # bus
        7 : 5, # truck
        80 : 3, # cyclist
    }

    annotation_sublist = []
    images_sublist = []

    annotation_index = -1
    image_index = -1
    if not os.path.exists('finished/images'):
        os.makedirs('finished/images')
    else:
        shutil.rmtree('finished/images')
        os.makedirs('finished/images')
        
    #for subdir in Path('output').glob('*'):
    for subdir in Path(args.dir).glob('*'):
        autolabel_path = Path(subdir) / 'voted_labels'
        stills_path = Path(subdir) / 'clean_images'
        
        for image_path in stills_path.glob('*'):
            
            image_index += 1
            width, height = imagesize.get(image_path)
            image_file_stem = image_path.stem
            
            #print('images' + image_path.name)
            images_sublist.append({'width':width,
                'height':height,
                'id':image_index,
                'file_name':('images/' + image_path.name)
                })
            
            label_path = Path(image_path.parent.parent) / 'voted_labels' / ( image_file_stem + '.txt' )
            shutil.copy(image_path, f'finished/images/{image_path.name}')
            with open(label_path, 'r') as label_file:
                for detection in label_file:
                    annotation_index += 1
                    # category, xywh, confidence
                    frame, track_id, x, y, w, h, confidence, category = detection.split(',')
                    new_category = autolabel_to_alphabetical_categories[int(category)]
                    yolo_bbox = BBox([float(x), float(y), float(w), float(h)], yolo)
                    coco_bbox = yolo_bbox.as_format(coco, width, height)
                    new_x = int(round(coco_bbox.value[0]))
                    new_y = int(round(coco_bbox.value[1]))
                    new_w = int(round(coco_bbox.value[2]))
                    new_h = int(round(coco_bbox.value[3]))
                    area = new_w * new_h
                    
                    annotation_sublist.append(
                        {'id': annotation_index,
                        'image_id': image_index,
                        'category_id':new_category,
                        'segmentation':[],
                        'bbox':[new_x, new_y, new_w, new_h],
                        'ignore':0,
                        'iscrowd':0,
                        'area': area
                        }
                    )
                    
    created_json.update(
        {'annotations': annotation_sublist}
    )
    created_json.update(
        {'images': images_sublist}
    )

    with open(os.path.abspath('finished/result.json'), 'w') as json_write:
        json.dump(created_json,json_write)

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Create a new json file from the autolabeled data.')
    parser.add_argument('--dir', type=str, default='output', help='Directory containing the autolabeled data.')
    args = parser.parse_args()
    run(args)