import cv2
import tensorflow as tf
import numpy as np

# Pre-trained TensorFlow model
model_path = 'saved_model'
model = tf.saved_model.load(model_path)

# Load image
image_path = 'image.jpg'
image = cv2.imread(image_path)

height, width, j = image.shape

# Preprocessing
input_tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_tensor = cv2.resize(input_tensor, (320, 320))
input_tensor = np.clip(input_tensor, 0, 255)  
input_tensor = np.expand_dims(input_tensor, axis=0)
input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)

# Object detection
detections = model(input_tensor)

# Extract detection results
boxes = detections['detection_boxes'][0].numpy()
scores = detections['detection_scores'][0].numpy()
classes = detections['detection_classes'][0].numpy()
num_detections = int(detections['num_detections'][0].numpy())

# Draw boundary
for i in range(num_detections):
    if scores[i] > 0.5:
        box = boxes[i] * [height, width, height, width]
        y1, x1, y2, x2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{int(classes[i])}: {scores[i]:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Result
cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()