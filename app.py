from flask import Flask, render_template, Response, jsonify
import BodyTrackingModule as btm
import HandTrackingModule as htm
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(1)
    body_detector = btm.poseDetector()
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)

        # Body Tracking
        img = body_detector.findPose(img)
        body_lmLists = body_detector.findPosition(img, draw=False)


        right_arm_raised = False
        left_arm_raised = False

        if len(body_lmLists) > 0:
            if body_lmLists[11][2] > body_lmLists[13][2] and body_lmLists[13][2] > body_lmLists[15][2]:
                cv2.circle(img, (body_lmLists[11][1], body_lmLists[11][2]), 10, (0,255,0), cv2.FILLED)
                cv2.circle(img, (body_lmLists[13][1], body_lmLists[13][2]), 10, (0,255,0), cv2.FILLED)
                cv2.circle(img, (body_lmLists[15][1], body_lmLists[15][2]), 10, (0,255,0), cv2.FILLED)
                right_arm_raised = True
            if body_lmLists[12][2] > body_lmLists[14][2] and body_lmLists[14][2] > body_lmLists[16][2]:
                cv2.circle(img, (body_lmLists[12][1], body_lmLists[12][2]), 10, (0,255,0), cv2.FILLED)
                cv2.circle(img, (body_lmLists[14][1], body_lmLists[14][2]), 10, (0,255,0), cv2.FILLED)
                cv2.circle(img, (body_lmLists[16][1], body_lmLists[16][2]), 10, (0,255,0), cv2.FILLED)
                left_arm_raised = True
        
        if right_arm_raised and left_arm_raised:
            cv2.circle(img, (body_lmLists[11][1], body_lmLists[11][2]), 10, (255,0,0), cv2.FILLED)
            cv2.circle(img, (body_lmLists[13][1], body_lmLists[13][2]), 10, (255,0,0), cv2.FILLED)
            cv2.circle(img, (body_lmLists[15][1], body_lmLists[15][2]), 10, (255,0,0), cv2.FILLED)

            cv2.circle(img, (body_lmLists[12][1], body_lmLists[12][2]), 10, (255,0,0), cv2.FILLED)
            cv2.circle(img, (body_lmLists[14][1], body_lmLists[14][2]), 10, (255,0,0), cv2.FILLED)
            cv2.circle(img, (body_lmLists[16][1], body_lmLists[16][2]), 10, (255,0,0), cv2.FILLED)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)