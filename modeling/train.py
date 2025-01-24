# modeling/train.py
from dataset import load_cifar10, extract_tensors
from modeling.predict import KnnClassifier
from config import K_VALUES, NUM_TEST_SAMPLES
import torch

def train_and_evaluate():
    # Загрузка данных
    train_set, test_set = load_cifar10()
    x_train, y_train = extract_tensors(train_set)
    x_test, y_test = extract_tensors(test_set)

    # Обучение модели
    clf = KnnClassifier()
    clf.fit(x_train, y_train)

    # Оценка модели
    for k in K_VALUES:
        res = clf.predict(x_test[:NUM_TEST_SAMPLES], k)
        print("K:", k, ", Acc:", clf.score(res, y_test[:NUM_TEST_SAMPLES]), "%")

    # Сохранение модели
    k = 1
    clf = KnnClassifier()
    clf.fit(x_train, y_train)
    result = clf.predict(x_test, k)
    print("-----------------------------------------------")
    print("K:", k, "acc:", clf.score(result, y_test), "%")

    # Экспорт модели в формат ONNX
    export_onnx(clf, x_test[:1])  # Экспортируем модель на одном примере

def export_onnx(model, sample_input, filename="models/knn_model.onnx"):
    """
    Экспортирует модель в формат ONNX.

    Параметры:
        model: Обученная модель.
        sample_input: Пример входных данных для модели.
        filename: Имя файла для сохранения ONNX-модели.
    """
    # Создаем директорию models, если она не существует
    import os
    os.makedirs("models", exist_ok=True)

    # Экспорт модели
    torch.onnx.export(
        model,  # Модель
        sample_input,  # Пример входных данных
        filename,  # Имя файла для сохранения
        opset_version=11,  # Версия ONNX
        input_names=["input"],  # Имя входного тензора
        output_names=["output"],  # Имя выходного тензора
        dynamic_axes={
            "input": {0: "batch_size"},  # Динамическая ось для входного тензора
            "output": {0: "batch_size"},  # Динамическая ось для выходного тензора
        },
    )
    print(f"Модель экспортирована в {filename}")

if __name__ == "__main__":
    train_and_evaluate()