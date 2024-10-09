from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse

def eval_json(ann_file, det_file):
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(det_file)
    cocoEval = COCOeval(coco_gt,coco_dt,'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    
    cocoEval.summarize()

    map50 = cocoEval.stats[1]

    map50 = round(float(map50), 5)

    return map50



def main():
    parser = argparse.ArgumentParser(description="Evaluate a detection file against an annotation file using COCO API.")
    
    parser.add_argument('--ann_file', type=str, 
                        default='test-instances_default.json', 
                        help="Path to the annotation file (e.g., instances_val.json).")
    parser.add_argument('--det_file', type=str, 
                        default='detections.json', 
                        help="Path to the detection result file (e.g., result.json).")
    
    args = parser.parse_args()
    
    # Run evaluation with the provided annotation and detection files
    map50 = eval_json(args.ann_file, args.det_file)
    
    print(f'mAP@50: {map50}')


if __name__ == "__main__":
    main()

