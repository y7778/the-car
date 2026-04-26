import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import cv2
import os

# 1. تحميل البيانات من ملف الـ CSV
# تأكد أن اسم الملف هو driving_log.csv
columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pd.read_csv('driving_log.csv', names=columns)

# 2. دالة معالجة الصور (يجب أن تكون نفس الموجودة في drive.py)
def img_preprocess(img_path):
    img = cv2.imread(img_path)
    img = img[60:135, :, :] # قص الصورة
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) # تغيير نظام الألوان
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66)) # تصغير الحجم لسرعة المعالجة
    img = img / 255 # توحيد قيم البكسلات
    return img

# 3. تجهيز مصفوفات الصور وزوايا التوجيه
images = []
steerings = []
for i in range(len(data)):
    # نأخذ صور الكاميرا الوسطى فقط كبداية
    img_path = data['center'][i]
    images.append(img_preprocess(img_path))
    steerings.append(float(data['steering'][i]))

X = np.array(images)
y = np.array(steerings)

# تقسيم البيانات لتدريب واختبار
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# 4. بناء نموذج NVIDIA المشهور للقيادة الذاتية
model = Sequential([
    Convolution2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'),
    Convolution2D(36, (5, 5), strides=(2, 2), activation='elu'),
    Convolution2D(48, (5, 5), strides=(2, 2), activation='elu'),
    Convolution2D(64, (3, 3), activation='elu'),
    Flatten(),
    Dense(100, activation='elu'),
    Dense(50, activation='elu'),
    Dense(10, activation='elu'),
    Dense(1) # مخرج واحد وهو زاوية التوجيه
])

model.compile(optimizer='adam', loss='mse')

# 5. بدء التدريب وحفظ النتيجة
print("بدأ التدريب... انتظر قليلاً")
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# حفظ الملف النهائي الذي ستحتاجه في الخطوة الرابعة
model.save('model.h5')
print("تم حفظ النموذج بنجاح باسم model.h5")

model.save('model.h5')