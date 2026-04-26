import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

# إعداد السيرفر والاتصال
sio = socketio.Server()
app = Flask(__name__)

# تحميل الموديل المدرب
model = load_model('model.h5')

def telemetry(sid, data):
    # استقبال الصورة من المحاكي
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = np.asarray(image)
    
    # معالجة الصورة لتناسب الموديل (Crop & Resize)
    image = image[60:135, :, :] 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.resize(image, (200, 66))
    image = image / 255.0 
    image = np.array([image])

    # توقع زاوية الدركسيون
    steering_angle = float(model.predict(image, batch_size=1))
    
    # تثبيت السرعة عند 15 كم/ساعة
    throttle = 0.15
    
    print(f'Steering: {steering_angle} | Throttle: {throttle}')
    send_control(steering_angle, throttle)

@sio.on('telemetry')
def on_telemetry(sid, data):
    if data:
        telemetry(sid, data)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

@sio.on('connect')
def connect(sid, environ):
    print("Connected to Simulator!")
    send_control(0, 0)

if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
