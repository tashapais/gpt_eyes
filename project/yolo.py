from ultralytics import YOLO
from PIL import Image
import cv2

# model = YOLO("yolov8n.pt")
# model.train(data='train.yaml', epochs=3)
# model('/Users/tashapais/Documents/Github/gpt_eyes/datasets/valid/images', show=True)
#
# model.save('/Users/tashapais/Documents/Github/gpt_eyes/project/weights.pt')

model = YOLO('/Users/tashapais/Documents/Github/gpt_eyes/ultralytics/runs/detect/train6/weights/best.pt')

im2 = cv2.imread("/Users/tashapais/Downloads/try.jpeg")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
#
# res = model("/Users/tashapais/Downloads/try.jpeg")
# res_plotted = res[0].plot(boxes="True", probs="True")
# cv2.imshow("result", res_plotted)
# results = model.predict("/Users/tashapais/Downloads/try.jpeg", show= True)
print(results)


while(True):
    pass

