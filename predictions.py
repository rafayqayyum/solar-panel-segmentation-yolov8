from ultralytics import YOLO
import cv2
import numpy as np
import argparse

# function takes an image, predicts the bounding boxes, saves the image with bounding boxes and returns the results
def predict(image,model):
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    masks = results[0].masks.masks.cpu().numpy()
    
    # draw bounding boxes
    for i in range(len(boxes)):
        box = boxes[i]
        conf = confs[i]
        cls = classes[i]
        mask = masks[i]
        if conf > 0.5:
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
            cv2.putText(image, str(cls), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(image, str(conf), (int(box[0]), int(box[1])+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            mask = np.array(mask*255, dtype=np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            ## resize mask to the size of the image
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            ## add red mask to the image
            image = cv2.addWeighted(image, 1, mask, 0.5, 0)
            
    cv2.imwrite("predictions.png", image)




if __name__=="__main__":
    # construct the argument parser for image path and model path
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-m", "--model", required=True, help="path to model")
    args = vars(ap.parse_args())
    model = YOLO(args["model"])  # initialize
    image=cv2.imread(args["image"])
    predict(image,model)
    
    
    