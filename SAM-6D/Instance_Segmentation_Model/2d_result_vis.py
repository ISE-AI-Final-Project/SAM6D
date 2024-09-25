import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
File structure
- 1 # Object
    - camera.json
    - obj_000005.ply
    - templates
        - mask_0.png
        - rgb_0.png
        - xyz_0.npy
        - mask_1.png
        - rgb_1.png
        - xyz_1.png
        - ...
    - 1 # Scene
        - depth.png
        - rgb.png
        - detection_ism.json
        - detection_pem.json
'''

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# s_x = str(input("Object : "))
# s_y = str(input("Scene : "))

file_path = '/home/icetenny/senior-1/SAM-6D/SAM-6D/Data/Example/outputs/sam6d_results/detection_ism.json'
# file_path = 'detection_ism.json'
json_data = read_json_file(file_path)

# rle: Dict[str, Any]
def rle_to_mask(rle) -> np.ndarray:
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order

for data in json_data:
    image = cv2.imread('/home/icetenny/senior-1/SAM-6D/SAM-6D/Data/Example/rgb.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bbox = data['bbox']
    score = data['score']
    rle = data['segmentation']
    # print('score = '+str(data['score']))
    # print('sem = '+str(data['sem_scores']))
    # print('appe = '+str(data['appe_scores']))
    # print('geo = '+str(data['geo_scores']))
    # print('visible = '+str(data['visible_ratio']))
    mask_img = rle_to_mask(rle)
    print(mask_img.shape)
    
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
    
    # Write the score on the top right of the bounding box
    score_text = f"{score:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (0, 255, 0)  # Green
    font_thickness = 2
    text_size, _ = cv2.getTextSize(score_text, font, font_scale, font_thickness)
    text_x = x + w - text_size[0]
    text_y = y - 10 if y - 10 > 10 else y + 10 + text_size[1]
    
    cv2.putText(image, score_text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].axis('off')
    
    axes[1].imshow(mask_img, cmap = 'gray')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    print('-------------------------------')