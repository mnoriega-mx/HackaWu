from flask import Flask, render_template, Response
import BodyTrackingModule as btm
import HandTrackingModule as htm
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLO model and classes
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    # Open video capture (use 1 for external camera or modify as needed)
    cap = cv2.VideoCapture(1)
    
    # Initialize body detector
    body_detector = btm.poseDetector()
    
    while True:
        success, img = cap.read()
        if not success:
            break

        # Flip image horizontally for a mirrored effect
        img = cv2.flip(img, 1)

        # Get the width and height of the frame
        height, width, channels = img.shape

        # Body Tracking
        img = body_detector.findPose(img)
        body_lmLists = body_detector.findPosition(img, draw=False)

        right_arm_raised = False
        left_arm_raised = False

        # Detect raised arms (example logic for detecting arms raised)
        if len(body_lmLists) > 0:
            if body_lmLists[11][2] > body_lmLists[13][2] and body_lmLists[13][2] > body_lmLists[15][2]:
                cv2.circle(img, (body_lmLists[11][1], body_lmLists[11][2]), 10, (0,255,0), cv2.FILLED)
                right_arm_raised = True
            if body_lmLists[12][2] > body_lmLists[14][2] and body_lmLists[14][2] > body_lmLists[16][2]:
                cv2.circle(img, (body_lmLists[12][1], body_lmLists[12][2]), 10, (0,255,0), cv2.FILLED)
                left_arm_raised = True
        
        if right_arm_raised and left_arm_raised:
            cv2.circle(img, (body_lmLists[11][1], body_lmLists[11][2]), 10, (255,0,0), cv2.FILLED)
            cv2.circle(img, (body_lmLists[12][1], body_lmLists[12][2]), 10, (255,0,0), cv2.FILLED)

        # Weapon Detection
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layer_names)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

        # Encode the image as JPEG and return the frame
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
