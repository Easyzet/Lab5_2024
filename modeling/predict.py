# modeling/predict.py
import torch
from tqdm import tqdm

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

    def score(self, y, y_ground):
        return int((torch.sum(torch.eq(y, y_ground)) / y.shape[0]) * 100)