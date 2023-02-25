from flask import Flask, send_file, render_template, request,Response
import replicate
import cv2, numpy as np, os, matplotlib.pyplot as plt, time, mediapipe as mp, keras.models as models, keras.layers as layers, keras.callbacks as callbacks
from numpy import load

app = Flask(__name__, static_folder='static')

@app.route('/sign')
def sign():
    return render_template('sign.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/text')
def text():
    return render_template('text.html')


def generate_frames():
        
    log_dir = os.path.join('Logs')
    tb_callback = callbacks.TensorBoard(log_dir=log_dir)
    actions = load('data.npy')
    model = models.Sequential([layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)), layers.LSTM(128, return_sequences=True, activation='relu'), layers.LSTM(64, return_sequences=False, activation='relu'), layers.Dense(64, activation='relu'), layers.Dense(32, activation='relu'), layers.Dense(actions.shape[0], activation='softmax')])
    model.load_weights('data')
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image.flags.writeable = False                  
        results = model.process(image)      
        image.flags.writeable = True                
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    def draw_styled_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(179, 55, 15), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(240, 242, 249), thickness=1, circle_radius=0.4)) 
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(7, 9, 193), thickness=1, circle_radius=0.4)) 
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(179, 55, 15), thickness=1, circle_radius=0.4)) 
    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose,lh, rh])
    array_data_sequence = []
    data_array_concate = []
    track = []
    threshold = 0.7
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            array_data_sequence.append(keypoints)
            array_data_sequence = array_data_sequence[-30:]
            if len(array_data_sequence) == 30:
                res = model.predict(np.expand_dims(array_data_sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                track.append(np.argmax(res))
                if np.unique(track[-20:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(data_array_concate) > 0 and actions[np.argmax(res)] != data_array_concate[-1]:
                            data_array_concate.append(actions[np.argmax(res)])
                        elif len(data_array_concate) == 0:
                            data_array_concate.append(actions[np.argmax(res)])
                if len(data_array_concate) > 5: 
                    data_array_concate = data_array_concate[-5:]
            cv2.rectangle(image,(384,120), (250,28), (0,0,255), 0)
            cv2.rectangle(image,(84,190), (150,78), (0,0,255), 0)
            cv2.rectangle(image,(549,190), (480,74), (0,0,255), 0)
            cv2.putText(image, 'Eye Area', (282,70), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.putText(image, 'Hand', (87,225), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.putText(image, 'Hand', (492,225), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.putText(image, ' '.join(data_array_concate), (3,460), cv2.FONT_HERSHEY_TRIPLEX , 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            ret,buffersssss=cv2.imencode('.jpg',image)
            frame=buffersssss.tobytes()

            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            if cv2.waitKey(10) & 0xFF == ord('s'):
                break
        cap.release()
        cv2.destroyAllWindows()






@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        width = int(request.form['width'])
        height = int(request.form['height'])
        prompt_strength = float(request.form['prompt_strength'])
        num_outputs = int(request.form['num_outputs'])
        num_inference_steps = int(request.form['num_inference_steps'])
        guidance_scale = float(request.form['guidance_scale'])
        model = replicate.models.get("stability-ai/stable-diffusion")
        version = model.versions.get("f178fa7a1ae43a9a9af01b833b9d2ecf97b1bcb0acfd2dc5dd04895e042863f1")

        inputs = {
            'prompt': prompt,
            'width': width,
            'height': height,
            'prompt_strength': prompt_strength,
            'num_outputs': num_outputs,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'scheduler': "DPMSolverMultistep",
        }

        output = version.predict(**inputs)
        image_url = output['outputs'][0]['output_url']

        return render_template('output.html', image_url=image_url)
    else:
        return render_template('home.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)