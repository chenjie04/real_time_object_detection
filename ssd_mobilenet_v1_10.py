import cv2
import argparse
import os
import numpy as np
import torch
import onnx
import onnxruntime


parser = argparse.ArgumentParser(description="Real time object detection with ssd_mobilenet_v1_10.")
parser.add_argument("-l","--labels",type=str,default="labels",help="path to labels used by object detection.")
parser.add_argument("-m","--models",type=str,default="models",help="path to object detection models.")
parser.add_argument("-d","--device",type=str,default="cpu",help="devices where to run the model.")
args = vars(parser.parse_args())

# Load the COCO class labels our model was trained on
labelsPath = os.path.sep.join([args["labels"],"coco.names"])
print(labelsPath)
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0,255,size=(len(LABELS),3),dtype="uint8")

# initialize Detection model
device = torch.device(args["device"])

ort_session = onnxruntime.InferenceSession("./models/ssd_mobilenet_v1_10.onnx")


# initialize the video stream, allow the cammera sensor to warmup.
print("[INFO] strating video stream.....")
webcam = cv2.VideoCapture(0,cv2.CAP_ANY) # Call the notebook built-in camera
# webcam = cv2.VideoCapture(0) # Call the notebook built-in camera
if not webcam.isOpened():
    print("[INFO] can't open the camera....")

else:
    # set the resolutions of camera, but it doesn't work on windows.
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH,512)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT,512)

    while True:
        # grab the frame from the threaded video stream.
        (grabed, frame) = webcam.read()
        height = frame.shape[0]
        width = frame.shape[1]
        
        frame = np.expand_dims(frame.astype(np.uint8), axis=0)
        # print(frame.shape)
        # compute ONNX Runtime output prediction
        # produce outputs in this order
        outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]

        result = ort_session.run(outputs, {"image_tensor:0": frame})

        num_detections, detection_boxes, detection_scores, detection_classes = result

        # print number of detections
        # print('num_detections.shape: ',num_detections.shape)
        # print(num_detections)
        # print('detection_class.shape: ',detection_classes.shape)
        # print(detection_classes)
        # print('detection_boxes.shape: ',detection_boxes.shape)
        # print(detection_boxes)

    
        # show the output frame
        frame = np.squeeze(frame)
        
        batch_size = num_detections.shape[0]
        for batch in range(0, batch_size):
            for detection in range(0, int(num_detections[batch])):
                c = detection_classes[batch][detection]
                confidence = detection_scores[batch][detection]
                d = detection_boxes[batch][detection]
                print(c)
                print(d)
                print('width,height:',width,height)
                # the box is relative to the image size so we multiply with height and width to get pixels.
                top = d[0] * height
                left = d[1] * width
                bottom = d[2] * height
                right = d[3] * width
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
                right = min(width, np.floor(right + 0.5).astype('int32'))
                print(left,top,right,bottom)
                color = [int(c_) for c_ in COLORS[int(c)]]
                text = "{}:{:.4f}".format(LABELS[int(c)-1],confidence)
                cv2.rectangle(frame,(left,top),(right,bottom),color,2)
                cv2.putText(frame,text,(left,top-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key was pressed, break frame the loop
        if key == ord("q"):
            break

webcam.release()
cv2.destroyAllWindows()        