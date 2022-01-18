# synthlabel
Generate accurate synthetic object detection labeled datasets, incorporating optional retroactive labeling and/or category fusion.

## Ingest Videos

All the input video is organized as:
```
VIDEO_DATA
├── front_autolabeled: copy outputs of autolabel.py to here
├── front_groundtruth: 
├── front_video: renamed videos (moved from RAW)
└── RAW: videos straight from camera (to be moved to 'front_video')
```
**Record new videos, then move them into `RAW`.**

**Rename videos to a standardized format:**

```
python3 video_renamer.py --dir <VIDEO_DATA>
```

## Inference Loop

```
python3 autolabel.py --dir=input_videos
```

+ YOLOv5 detects objects
+ IOU tracker tracks objects
+ Category Fusion merges object categories

## Validate Output

Sometimes, empty or corrupt files come out of the inference loop. Probably high energy cosmic rays or something. Fix it with this.

```
python3 validate_autolabels.py --dir=output
```

## Retrolabeling Step

Once all the labels have been produced and saved, the retrolabeling step goes through the labels (not images) to refine the categories.

```
python3 retrolabel.py --dir=output
```

## Option A: Skip human labeling and prepare a synthetic dataset

**Convert auto- & retrolabel output to COCO formatted dataset**

```
python3 skip_handlabel.py
```

Outputs are saved to `label-studio/finished` directory.

**Compress contents of `finished`**:
```
cd label-studio/finished
zip -r <NAME_OF_DATASET>.zip images result.json
```
## Option B: Import into Label Studio for Assisted Human Labeling

**Install label-studio:**
```
pip3 install label-studio
```

**Create label-studio.json labeling instruction file**:
```
python3 label-studio/generate_labelstudio_task.py
```

**Start HTTP file server locally**:
```
label-studio/serve_local_file.sh output
```

**Start Label Studio web GUI**:
```
label-studio
```

**Create Project with Custom Template:**

Copy the following HTML to Label Studio custom template:

```html
<View>
  <View style="display:flex;align-items:start;gap:8px;flex-direction:row-reverse">
    <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
    <View>
      <Filter toName="label" minlength="0" name="filter"/>
      <RectangleLabels name="label" toName="image" showInline="false">
        <Label value="cyclist" background="#FFA3FF"/>
        <Label value="pedestrian" background="#D4380D"/>
        <Label value="car" background="#FFC069"/>
        <Label value="truck" background="#FFFF00"/>
        <Label value="bus" background="#D3F261"/>
        <Label value="motorcycle" background="#0062ff"/>
        <Label value="bicycle" background="#FFA39E"/>
      </RectangleLabels>
    </View>
  </View>
</View>
```

**`Import` the `labelstudio-task.json` file in `output` directory.**

**Do annotation work**

**Export -> COCO**

## Combine train and val dataets for YOLOX training

Use `labelstudio_coco_to_real_coco.sh` to reorganize the directory structure to be exactly the same as real COCO dataset. We avoid having to write new PyTorch dataloaders this way.

+ Prepare two datasets (one for train, one for val) and export them separately.
+ Load them into the converter as first and second arguments as follows:

```
label-studio/labelstudio_coco_to_real_coco.sh TRAIN.zip VAL.zip
```

**Place the resulting `COCO` directory into `YOLOX/datasets`. See [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) for next step, which is training.**



