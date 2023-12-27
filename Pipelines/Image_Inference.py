import sys
import torch
import os
import matplotlib.pyplot as plt
import cv2

model = torch.hub.load('../yolov5',
                       'custom', path='../yolov5/runs/train/exp/weights/best.pt',
                       source='local', device='cuda:0')

image_file = sys.argv[1]

results = model(image_file, size=640)

df = results.pandas().xyxy[0]

color_dict = {
    0: (255, 0, 0),    # Red
    1: (0, 255, 0),    # Green
    2: (0, 0, 255),    # Blue
    3: (200, 200, 0),  # Dark Yellow
    4: (0, 255, 255),  # Cyan
    5: (255, 0, 255),  # Magenta
    6: (128, 0, 128),  # Purple
    7: (255, 165, 0)   # Orange
}

image = cv2.imread(image_file)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

for ind in df.index:
    xmin = int(df['xmin'][ind])
    ymin = int(df['ymin'][ind])
    xmax = int(df['xmax'][ind])
    ymax = int(df['ymax'][ind])
    classNo = int(df['class'][ind])

    # bounding box
    img = cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), color_dict[classNo], 2)

for ind in df.index:
    xmin = int(df['xmin'][ind])
    ymin = int(df['ymin'][ind])
    xmax = int(df['xmax'][ind])
    ymax = int(df['ymax'][ind])
    confidence = df['confidence'][ind]
    classNo = int(df['class'][ind])
    name = df['name'][ind]

    # text size and width
    (text_width, text_height) = cv2.getTextSize(f'{name} {confidence:.2f}', cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

    # putting background
    cv2.rectangle(image_rgb, (xmin-2,ymin), (xmin+text_width-2, ymin-text_height-4), color_dict[classNo], cv2.FILLED)
    
    # object name
    cv2.putText(image_rgb, f'{name} {confidence:.2f}', (xmin, ymin-2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)


filename = image_file.split('/')[-1]

text_filename,_ = filename.split('.')


# save image with bboxes
cv2.imwrite(f'Inferences/images/{filename}', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

# save predictions of model
df.to_csv(f'Inferences/labels/{text_filename}.txt', sep='\t', index=False, header=False)
