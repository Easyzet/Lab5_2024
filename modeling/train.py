# modeling/train.py
from dataset import load_cifar10, extract_tensors
from modeling.predict import KnnClassifier
from config import K_VALUES, NUM_TEST_SAMPLES

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

if __name__ == "__main__":
    train_and_evaluate()