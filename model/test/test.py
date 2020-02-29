import os
import csv
from model.mask import Mask

mask = Mask(
    facedet_model="/Users/rainweic/DeepLearning/MaskDet-opensource/params",
    facecla_model="/Users/rainweic/DeepLearning/MaskDet-opensource/params/best-0.889.params",
    use_gpu=False
)

mask.load_model()

test_jpg_root = "/Users/rainweic/Downloads/mask/facedet"

test_jpgs = os.listdir(test_jpg_root)
test_jpgs = sorted(test_jpgs)
test_jpgs = [os.path.join(test_jpg_root, i) for i in test_jpgs]
# print(test_jpgs)

f = open("result.csv",'w',encoding='utf-8')
csv_writer = csv.writer(f)

for jpg in test_jpgs:
    print(jpg)
    result = mask.get_result(jpg)
    print(result)

    img_id = os.path.basename(jpg)[:-4]
    face_without_mask = result['face_without_mask']
    face_with_mask = result['face_with_mask']

    csv_writer.writerow([img_id, face_without_mask, face_with_mask])

f.close()