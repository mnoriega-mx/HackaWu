from flask import Flask, render_template, Response, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import BodyTrackingModule as btm
import HandTrackingModule as htm
import cv2
import numpy as np
import time
import pytz
from datetime import datetime
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///logs.db'
db = SQLAlchemy(app)

# Twilio credentials
account_sid = 'text'  # Replace with your Account SID
auth_token = 'text'  # borrarlo cuando se haga commits
client = Client(account_sid, auth_token)

twilio_phone_number = '+number'  # Replace with your Twilio phone number
your_phone_number = '+number'  # Replace with your personal phone number

# Load YOLO model and classes
net = cv2.dnn.readNet("env/yolov3_training_2000.weights", "env/yolov3_testing.cfg")
classes = ["Weapon"]
output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

class Logs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    log = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Function to save log
last_log_time = time.time()
log_interval = 5  # Minimum time (in seconds) between logs

def save_log(log_message):
    global last_log_time
    current_time = time.time()
    
    if current_time - last_log_time >= log_interval:
        with app.app_context():
            new_log = Logs(log=log_message)
            db.session.add(new_log)
            db.session.commit()

        # Send SMS alert for specific logs
        if "Raised arms" in log_message or "Weapon" in log_message:
            send_sms_alert(log_message)

        last_log_time = current_time  # Update the last log time

# Function to send SMS alert via Twilio
def send_sms_alert(log_message):
    client.messages.create(
        body=f"Alert: {log_message}",
        from_=twilio_phone_number,
        to=your_phone_number
    )

@app.route('/')
def index():
    # Fetch all logs to display on the page
    logs = Logs.query.all()
    return render_template('index.html', logs=logs)

@app.route('/logs')
def get_logs():
    local_timezone = pytz.timezone('America/Mexico_City')  # Replace with your desired time zone
    logs = Logs.query.all()
    logs_data = [{'time': log.timestamp.replace(tzinfo=pytz.utc).astimezone(local_timezone).strftime('%H:%M:%S'), 'log': log.log} for log in logs]
    return jsonify(logs_data)

def generate_frames():
    arms_raised = False
    weapon = False
    # Open video capture (use 1 for external camera or modify as needed)
    cap = cv2.VideoCapture(0)
    
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
            # Log event
            if not arms_raised:
                with app.app_context():
                    save_log("Raised arms")
                arms_raised = True
        else:
            arms_raised = False

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
                # Log weapon detection
                if not weapon:
                    with app.app_context():
                        save_log("Weapon")
                    weapon = True
                else:
                    weapon = False

        # Encode the image as JPEG and return the frame
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to handle incoming SMS and forward them to your phone
@app.route("/sms", methods=['POST'])
def sms_reply():
    # Get the message the user sent
    body = request.form['Body']
    from_number = request.form['From']

    # Forward the message to your personal phone number
    client.messages.create(
        body=f"Message from {from_number}: {body}",
        from_=twilio_phone_number,
        to=your_phone_number
    )

    # Respond to the sender
    resp = MessagingResponse()
    resp.message("Your message has been forwarded.")
    return str(resp)

if __name__ == "__main__":
    with app.app_context():
        db.drop_all()  # Drop all tables, including the 'logs' table
        db.create_all()  # Recreate the tables with the new schema
        Logs.query.delete()  # Clear all logs from the previous session
        db.session.commit()  # Commit the changes
    app.run(debug=True)
