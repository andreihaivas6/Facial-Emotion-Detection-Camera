import cv2
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np

'''
Tutorial: https://www.youtube.com/watch?v=mj-3vzJ4ZVw&ab_channel=HackersRealm
Dataset: https://www.kaggle.com/datasets/aadityasinghal/facial-expression-dataset
Drive for Google Colab: https://drive.google.com/drive/folders/1A8pA1vw8escCyjd4rjUzIVg9ZKpSTS7U
'''

model = load_model('model.h5', compile=False)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)
while True:
    try:
        ret, test_img = capture.read()
        if not ret:
            continue
        image = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        faces_detected = face_detector.detectMultiScale(image, 1.32, 5)
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)

            image = image[y:y + w, x:x + h]
            image = cv2.resize(image, (48, 48))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            plt.imshow(image, cmap='gray')
            plt.show()

            image = np.array(image)
            image = image / 255.0
            image = image.reshape(1, 48, 48, 1)

            result_values = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

            predicted = model.predict(image)
            index_of_result = np.argmax(predicted)
            result = result_values[index_of_result]

            cv2.putText(test_img, result, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ', resized_img)

        if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
            break
    except:
        pass
