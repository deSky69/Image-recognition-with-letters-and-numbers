import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from emnist import extract_training_samples, extract_test_samples
from tensorflow.keras.utils import to_categorical
import pyautogui

def extract_and_save_characters(image_path, output_folder):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresholded = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # Сортировка по координате X
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Отфильтровываем слишком маленькие контуры
          
            # Добавляем отступ
            x -= 20
            y -= 20
            w += 40
            h += 40
            
            # Убедимся, что координаты не выходят за пределы изображения
            x = max(x, 0)
            y = max(y, 0)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)
            
            character = img[y:y+h, x:x+w]
            filename = os.path.join(output_folder, f"{idx}.png")
            cv2.imwrite(filename, character)

# Загрузка данных EMNIST Letters и EMNIST Digits
images_train_letters, labels_train_letters = extract_training_samples('letters')
images_test_letters, labels_test_letters = extract_test_samples('letters')

images_train_digits, labels_train_digits = extract_training_samples('digits')
images_test_digits, labels_test_digits = extract_test_samples('digits')

# Объединение данных букв и цифр
images_train_combined = np.concatenate((images_train_letters, images_train_digits), axis=0)
labels_train_letters_shifted = labels_train_letters - 1  # Смещение меток букв
labels_train_combined = np.concatenate((labels_train_letters_shifted, labels_train_digits + 26), axis=0)

images_test_combined = np.concatenate((images_test_letters, images_test_digits), axis=0)
labels_test_letters_shifted = labels_test_letters - 1  # Смещение меток букв
labels_test_combined = np.concatenate((labels_test_letters_shifted, labels_test_digits + 26), axis=0)

# Количество классов
num_classes_combined = 36 

# Преобразование данных к нужному формату
images_train_combined = images_train_combined.astype('float32') / 255.0
images_test_combined = images_test_combined.astype('float32') / 255.0

images_train_combined = np.expand_dims(images_train_combined, axis=-1)
images_test_combined = np.expand_dims(images_test_combined, axis=-1)

# One-hot кодирование меток
labels_train_onehot_combined = to_categorical(labels_train_combined, num_classes=num_classes_combined)
labels_test_onehot_combined = to_categorical(labels_test_combined, num_classes=num_classes_combined)

# Определение архитектуры модели
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes_combined, activation='softmax'))


# Создание окна для вывода на экран букв,цифр
screen_width, screen_height = pyautogui.size()

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(images_train_combined, labels_train_onehot_combined, epochs=10, batch_size=64, validation_split=0.2)

# Оценка точности на тестовых данных
loss, accuracy = model.evaluate(images_test_combined, labels_test_onehot_combined)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Сохранение модели
model.save('emnist_combined_model.h5')

# Загрузка модели
model = tf.keras.models.load_model('emnist_combined_model.h5')

digit_number = 1

while os.path.isfile(f'D:\\aaa\\{digit_number}.png'):

    img_path = f'D:\\aaa\\{digit_number}.png'
    output_folder = f'D:\\bbb'
    
    extract_and_save_characters(img_path, output_folder)
    
    digit_number += 1
    
digit_number = 0

while os.path.isfile(f'D:\\bbb\\{digit_number}.png'):
    img = cv2.imread(f'D:\\bbb\\{digit_number}.png', cv2.IMREAD_GRAYSCALE)
    img = 255 - img
    img = cv2.resize(img, (28, 28))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    if predicted_class_index < 26:
        predicted_class = chr(ord('A') + predicted_class_index)
    else:
        predicted_class = str(predicted_class_index - 26)  # Обновленное преобразование для цифр
    print(f"Предсказанный символ: {predicted_class}\n")
    
    probabilities = prediction[0]
    for class_index, probability in enumerate(probabilities[:-1]):
        if class_index < 25:
            label = chr(ord('A') + class_index)
        else:
            label = str(class_index - 25)  # Обновленное преобразование для цифр
        print(f"Вероятность для {label}: {probability:.4f}")
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img[0, :, :, 0], cmap=plt.cm.binary)
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+{}+{}".format(screen_width - 400, screen_height - 500))
    plt.show()
    digit_number += 1
