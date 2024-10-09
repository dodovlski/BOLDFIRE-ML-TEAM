import json
from pycocotools.coco import COCO

annotation_file_path = 'datathon/val/annotations/val.json'

coco_ann = COCO(annotation_file=annotation_file_path)
imgfile2imgid = {coco_ann.imgs[i]['file_name']: i for i in coco_ann.imgs.keys()}

with open('image_file_name_to_image_id.json', 'w') as f:

   json.dump(imgfile2imgid, f)