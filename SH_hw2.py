import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

class SimpleNeuralNetwork:
    def __init__(self, input_size=784, hidden_size1=1176, hidden_size2=784, output_size=10):
        # Инициализация весов случайным образом
        self.W1 = np.random.uniform(-0.5, 0.5, (hidden_size1, input_size))
        self.W2 = np.random.uniform(-0.5, 0.5, (hidden_size2, hidden_size1))
        self.W3 = np.random.uniform(-0.5, 0.5, (output_size, hidden_size2))  # 10 классов

    @staticmethod
    def activation_function(x):
        return 2 / (1 + np.exp(-x)) - 1

    @staticmethod
    def activation_derivative(x):
        return 0.5 * (1 + x) * (1 - x)

    def go_forward(self, inp):
        sum1 = np.dot(self.W1, inp)
        out1 = np.array([self.activation_function(x) for x in sum1])

        sum2 = np.dot(self.W2, out1)
        out2 = np.array([self.activation_function(x) for x in sum2])

        sum3 = np.dot(self.W3, out2)
        y = self.activation_function(sum3)
        return y, out1, out2

    def train(self, X_train, y_train, learning_rate=0.01, epochs=100):
        count = len(X_train)
        
        for k in range(epochs):
            x = X_train[k]  
            y_true = np.zeros(10)  # создаем вектор для истинных значений
            y_true[y_train[k]] = 1  # устанавливаем единицу в позиции класса
            
            y, out1, out2 = self.go_forward(x[0:784])  # прямой проход по НС

            e = y - y_true                           # ошибка
            delta = e * self.activation_derivative(y)  # локальный градиент для выхода

            self.W3 -= learning_rate * delta[:, np.newaxis] * out2[np.newaxis, :]  # корректировка весов выходного слоя

            delta2 = self.W3.T @ delta * self.activation_derivative(out2)          # локальный градиент второго скрытого слоя
            self.W2 -= learning_rate * np.outer(delta2, out1)      # корректировка весов второго скрытого слоя

            delta1 = self.W2.T @ delta2 * self.activation_derivative(out1)       # локальный градиент первого скрытого слоя
            self.W1 -= learning_rate * np.outer(delta1, x[0:784])  # корректировка весов первого слоя

    def predict(self, X):
        predictions = []
        for x in X:
            y, _, _ = self.go_forward(x[0:784])
            predicted_class = np.argmax(y)  # класс с максимальным значением
            predictions.append(predicted_class)
        return np.array(predictions)
    

# Пример использования класса
if __name__ == "__main__":
    # Создаем случайные данные для примера

    # Загружаем данные из https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', return_X_y = True, as_frame = False)

    y = y.astype('int8')

    N_train = 6000
    N_test = 1000

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size = N_train, test_size = N_test, 
                                                    stratify = y, random_state = 42)
    
    # Инициализация и обучение нейронной сети
    nn = SimpleNeuralNetwork()
    nn.train(X_train, y_train)

    predictions = nn.predict(X_test)

    for i in range(100):
        print(f"Предсказанный класс: {predictions[i]}, Истинный класс: {y_test[i]}")