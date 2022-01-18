# synthlabel
Generate accurate synthetic object detection labeled datasets, incorporating optional retroactive labeling and/or category fusion.

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


