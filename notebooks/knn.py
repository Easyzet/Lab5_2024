import torch
from tqdm import tqdm
from torchvision import datasets
import matplotlib.pyplot as plt

#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

# Загрузка обучающей и тестовой выборки
train_set = datasets.CIFAR10(
    root="data",
    train=True,
    download=True
)

test_set = datasets.CIFAR10(
    root="data",
    train=False,
    download=True
)

# Список классов этого датасета
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def extract_tensors(dataset):
  x = torch.tensor(dataset.data, dtype=torch.float32).permute(0, 3, 1, 2).div_(255)
  y = torch.tensor(dataset.targets, dtype=torch.int8)
  return x, y

# С помощью функции extract_tensors преобразуйте train_set и test_set.

(x_train, y_train) = extract_tensors(train_set)
(x_test, y_test) = extract_tensors(test_set)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

index = 0
img = x_train[index]
numClass = y_train[index]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(img[0])  # отображение канала R
ax1.set_title("R")
ax2.imshow(img[1])  # отображение канала G
ax2.set_title("G")
ax3.imshow(img[2])  # отображение канала B
ax3.set_title("B")
#fig.show()

print(classes[numClass])  # вывод класса

class KnnClassifier:

    def fit(self, X, y):
      self.Xtr = X
      self.ytr = y

    def predict(self, samples, k):
        Ypred = torch.zeros(samples.shape[0])
        for i, sample in tqdm(enumerate(samples), total=samples.shape[0], desc="Predicting"):
            sample = sample.unsqueeze(0)
            distance = torch.sum((self.Xtr - sample)**2, 1)**0.5
            distance = distance.mean(dim=(1, 2))
            kmeans = torch.topk(distance, k, sorted=True, largest=False).indices.mode().values
            Ypred[i] = self.ytr[kmeans]
        return Ypred
    
clf = KnnClassifier()
clf.fit(x_train, y_train)  # Обучите вашу модель

countOfTests = 50
res = clf.predict(x_test[:countOfTests], 3)  # Проверьте ее предсказания

def score(y, y_ground):
  return int((torch.sum(torch.eq(y, y_ground))/y.shape[0])*100)

score(res, y_test[:countOfTests])  # Оцените качество работы вашей моделиq

# реализуйте подбор параметра k

K = [1,3,5,7,9,11,15,20,30,40,50,60]
for k in K:
    res = clf.predict(x_test[:countOfTests], k)
    print("K:", k, ", Acc:", score(res, y_test[:countOfTests]),"%")

k=1
clf = KnnClassifier()
clf.fit(x_train, y_train)
result = clf.predict(x_test, k)
print("-----------------------------------------------")
print("K:", k, "acc:", score(result, y_test),"%")