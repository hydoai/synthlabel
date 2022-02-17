# USAGE
# 
# ./labelstudio_coco_to_real_coco [TRAIN_COCO.zip] [VAL_COCO.zip]
# args:
#     $1: path to zipped TRAIN dataset (direct output from label studio (COCO))
#     $2: path to zipped VALIDATION dataset (direct output from label studio (COCO))

mkdir COCO
mkdir COCO/unzipped_train
mkdir COCO/unzipped_val 

tar -xvf $1 -C COCO/unzipped_train
tar -xvf $2 -C COCO/unzipped_val

cd COCO

mkdir annotations train2017 val2017

mv unzipped_train/result.json annotations/instances_train2017.json
mv unzipped_val/result.json annotations/instances_val2017.json

mv unzipped_train/images train2017
mv unzipped_val/images val2017

rmdir unzipped_train
rmdir unzipped_val

