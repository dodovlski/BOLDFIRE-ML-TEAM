import json
import cv2
import os

test_images_path = 'datathon/test/test_images'

image_file_name_to_image_id = json.load(open('image_file_name_to_image_id.json'))

results = []
for img_name in os.listdir(test_images_path):
   image = cv2.imread(os.path.join(test_images_path, img_name))

   #  ----------------- This part includes the pre-process and model inference -----------------  #

   data = pre_process(image)
   bboxes, labels, scores = model(data)

   #  ----------------- This part includes the pre-process and model inference -----------------  #

   img_id = image_file_name_to_image_id[img_name]
   for bbox, label, score in zip(bboxes, labels, scores):
       bbox[2], bbox[3] = bbox[2] - bbox[0], bbox[3] - bbox[1] # xyxy to xywh
       res = {

           'image_id': img_id,
           'category_id': int(label) + 1, # add 1 to label if your model output starts from 0, the labels in the dataset starts from 1
           'bbox': list(bbox.astype('float64')),
           'score': float("{:.8f}".format(score.item()))
       }
       results.append(res)

with open('your_name.json', 'w') as f:
   json.dump(results, f)