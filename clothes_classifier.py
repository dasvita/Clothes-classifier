import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np



# x_train - матрица, в которой каждая строка – это отдельный объект, а каждый столбец – это отдельный признак объекта
# y_train – вектор меток объектов (его длина должна совпадать с количеством строк в матрице)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Список с названиями классов
classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

# Преобразование размерности и нормализация данных в наборе
x_train = x_train.reshape(60000, 784)
x_train = x_train / 255

# Преобразование меток в формат one hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)

# Создание последовательной модели
model = Sequential()

# Добавление слоёв сети, всего два полносвязных слоя
# Входной полносвязный слой, 800 нейронов, 784 входа в каждый нейрон
model.add(Dense(800, input_dim=784, activation="relu"))
# Выходной полносвязный слой, 10 нейронов
model.add(Dense(10, activation="softmax"))


# metrics - метрика, на основе которой мы делаем вывод о качестве модели
# loss - функция потерь, на основе которой нейронная сеть понимает на сколько она ошиблась
# optimizer - функция, которая реализует определенную стратегию обратного распространения ошибки по весам нейронной сети

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Создаем лист с обратными связями для избежания черезменого обучения
callbacks_list = [EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True),
                 ]


# batch_size - количество объектов из матрицы x_train, которые подаются на вход нейронной сети за одну эпоху
# epochs – количество эпох, на протяжении которых будет учиться нейронная сеть
# verbose – отвечает за выведение информации о процессе обучения

model.fit(x_train, y_train,
        batch_size=200,
        epochs=5,
        callbacks=callbacks_list,
        verbose=1)


# Запускаем распознавание набора данных, на котором обучалась сеть
predictions = model.predict(x_train)

# Номер изображения для распознавания, меняйте значение "n" для распознавания других изображений
n = 0

# Выводим название класса, который предлагает модель
print("Класс изображения определён как - " + classes[np.argmax(predictions[n])])

# Выводим истинное название класса
print("Истинный класс изображения - " + classes[np.argmax(y_train[n])])

if classes[np.argmax(predictions[n])] == classes[np.argmax(y_train[n])]:
        print("Класс определён верно! ;)")
else:
        print ("Класс определён неверно! :(")
