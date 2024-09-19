import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, num_classes=4):
        super(Attention, self).__init__()
        self.num_classes = num_classes
        self.M = 500
        self.L = 128

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),
            nn.Linear(self.L, 1)  # vector w
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        N, K, C, H, W = x.shape

        x = x.contiguous().view(N * K, C, H, W)  # (N x K) x C x H x W

        H = self.feature_extractor_part1(x)
        H = H.contiguous().view(N * K, -1)
        H = self.feature_extractor_part2(H)  # (N x K) x M

        A = self.attention(H)  # (N x K) x 1
        A = A.contiguous().view(N, 1, K)  # N x 1 x K
        A = F.softmax(A, dim=2)  # softmax over K

        H = H.contiguous().view(N, K, self.M)  # N x K x M
        Z = torch.matmul(A, H)  # N x 1 x M
        Z = Z.squeeze(1)  # N x M

        y_proba = self.classifier(Z)  # N x num_classes
        y_hat = torch.argmax(y_proba, dim=1)  # N

        return y_proba, y_hat, A.squeeze(1)


class Additive(nn.Module):
    def __init__(self, num_classes=4):
        super(Additive, self).__init__()
        self.num_classes = num_classes
        self.M = 500
        self.L = 128

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),
            nn.Linear(self.L, 1)  # vector w
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, self.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, K, C, H, W = x.shape

        x = x.contiguous().view(N * K, C, H, W)  # (N x K) x C x H x W

        H = self.feature_extractor_part1(x)
        H = H.contiguous().view(N * K, -1)
        H = self.feature_extractor_part2(H)  # (N x K) x M

        A = self.attention(H)  # (N x K) x 1
        A = A.contiguous().view(N, K, 1)  # N x K x 1
        A = F.softmax(A, dim=1)  # softmax over K

        H = H.contiguous().view(N, K, self.M)  # N x K x M
        Z = torch.mul(A, H)  # N x K x M
        Z = Z.contiguous().view(N * K, self.M)  # (N x K) x M

        P = self.classifier(Z)  # (N x K) x num_classes
        P = P.contiguous().view(N, K, self.num_classes)  # N x K x num_classes

        y_proba = torch.mean(P, dim=1)  # N x num_classes
        y_hat = torch.argmax(y_proba, dim=1)  # N

        return y_proba, y_hat, A.squeeze(2), P


if __name__ == '__main__':
    X = torch.rand(4, 16, 1, 28, 28)

    model = Attention()
    y_proba, y_hat, A = model(X)
    print(y_proba.shape, y_hat.shape, A.shape)

    model = Additive()
    y_proba, y_hat, A, P = model(X)
    print(y_proba.shape, y_hat.shape, A.shape, P.shape)