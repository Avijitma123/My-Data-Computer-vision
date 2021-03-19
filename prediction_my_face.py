import cv2
import numpy as np
from PIL import Image
from keras import models
from mtcnn import MTCNN
detector = MTCNN()

#Load the saved model
model = models.load_model('model3.h5')
video = cv2.VideoCapture(0)
class_lable=['Avijit','Bakul']

while True:
        ret, frame = video.read()
        """frame = cv2.resize(frame, (600, 400))
        boxes = detector.detect_faces(frame)
        if boxes:
 
             box = boxes[0]['box']
             conf = boxes[0]['confidence']
             x, y, w, h = box[0], box[1], box[2], box[3]
 
             if conf > 0.5:
                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)"""

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into 128x128 because we trained the model with this image size.
        im = im.resize((128,128))
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict 'me' on the image
        prediction = model.predict(img_array)[0][0]
        if(prediction==1):
            print("Avijit Dey")

        #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
        elif prediction == 0:
                print("Bakul Dey")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(prediction)
        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()