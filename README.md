# synthlabel
Generate accurate synthetic object detection labeled datasets, incorporating optional retroactive labeling and/or category fusion.

## Architecture

### Inference Loop

+ YOLOv5 detects objects
+ IOU tracker tracks objects
+ Category Fusion merges object categories

### Retrolabeling Step

Once all the labels have been produced and saved, the retrolabeling step goes through the labels (not images) to refine the categories.

```
python3 retrolabel.py --dir=<path_to_output_dir>
```


