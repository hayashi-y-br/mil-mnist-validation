import os
import sys

from omegaconf import DictConfig, open_dict
import hydra
import numpy as np
import torch
import torch.optim as optim
from torchvision.utils import make_grid

from dataset import MyDataset
from model import Attention, Additive


class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='model_weights.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model = None

    def __call__(self, valid_loss, model):
        if valid_loss < self.best_loss - self.delta:
            self.best_loss = valid_loss
            self.counter = 0
            self.best_model = model.state_dict()
            torch.save(self.best_model, self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print('Early stopping')
                return True
        return False

def save_img(X, path='./img/', filename='img', nrow=4, mean=torch.tensor([0.5]), std=torch.tensor([0.5])):
    X = make_grid(X, nrow=nrow, padding=0)[0]
    X = X * std + mean
    np.savetxt(path + filename, X.numpy(), delimiter=',')


def save_score(S, path=f'./score/', filename='score', nrow=4):
    S = S.contiguous().view(nrow, nrow)
    np.savetxt(path + filename, S.numpy(), delimiter=',')


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    sys.stdout = open('stdout.txt', 'w')
    os.makedirs('img')
    os.makedirs('score')

    with open_dict(cfg):
        cfg.use_cuda = cfg.use_cuda and torch.cuda.is_available()

    torch.manual_seed(cfg.seed)
    if cfg.use_cuda:
        print(torch.cuda.get_device_name())
        torch.cuda.manual_seed(cfg.seed)

    print('Load Train, Validation and Test Set')
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if cfg.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        MyDataset(train=True, valid=False, **cfg.dataset),
        batch_size=cfg.settings.batch_size,
        shuffle=True,
        **loader_kwargs
    )
    valid_loader = torch.utils.data.DataLoader(
        MyDataset(train=True, valid=True, **cfg.dataset),
        batch_size=1,
        shuffle=False,
        **loader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        MyDataset(train=False, valid=False, **cfg.dataset),
        batch_size=1,
        shuffle=False,
        **loader_kwargs
    )

    print('Init Model')
    if cfg.model.name == 'attention':
        model = Attention()
    elif cfg.model.name == 'additive':
        model = Additive()
    if cfg.use_cuda:
        model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.reg)
    early_stopping = EarlyStopping(patience=cfg.settings.patience, delta=cfg.settings.delta, path=cfg.path)

    print('Start Training')
    train_loss_list = []
    train_accuracy_list = []
    valid_loss_list = []
    valid_accuracy_list = []
    for epoch in range(1, cfg.settings.epochs + 1):
        # Training
        model.train()
        train_loss = 0.
        train_accuracy = 0.
        for i, (X, y) in enumerate(train_loader):
            if cfg.use_cuda:
                X, y = X.cuda(), y.cuda()

            optimizer.zero_grad()

            y_proba, y_hat, *_ = model(X)

            loss = loss_fn(y_proba, y)
            loss.backward()

            optimizer.step()

            train_loss += loss.detach().cpu().item()
            train_accuracy += y_hat.eq(y).detach().cpu().mean(dtype=float)
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        print('Epoch: {:2d}, Training Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch, train_loss, train_accuracy), end=', ')

        # Validation
        model.eval()
        valid_loss = 0.
        valid_accuracy = 0.
        with torch.no_grad():
            for i, (X, y) in enumerate(valid_loader):
                if cfg.use_cuda:
                    X, y = X.cuda(), y.cuda()

                y_proba, y_hat, *_ = model(X)
                loss = loss_fn(y_proba, y)

                valid_loss += loss.detach().cpu().item()
                valid_accuracy += y_hat.eq(y).detach().cpu().mean(dtype=float)
        valid_loss /= len(valid_loader)
        valid_accuracy /= len(valid_loader)
        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)
        print('Validation Loss: {:.4f}, Accuracy: {:.4f}'.format(valid_loss, valid_accuracy))
        if early_stopping(valid_loss, model):
            break
    train_loss = np.array(train_loss_list)
    train_accuracy = np.array(train_accuracy_list)
    valid_loss = np.array(valid_loss_list)
    valid_accuracy = np.array(valid_accuracy_list)
    np.savetxt('train_loss.csv', train_loss, delimiter=',')
    np.savetxt('train_accuracy.csv', train_accuracy, delimiter=',')
    np.savetxt('validation_loss.csv', valid_loss, delimiter=',')
    np.savetxt('validation_accuracy.csv', valid_accuracy, delimiter=',')

    print('Start Testing')
    y_list = []
    y_hat_list = []
    model.load_state_dict(torch.load(cfg.path, weights_only=True))
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    with torch.no_grad():
        num_classes = cfg.model.num_classes
        for i, (X, y) in enumerate(test_loader):
            if cfg.use_cuda:
                X, y = X.cuda(), y.cuda()

            y_proba, y_hat, *score = model(X)
            loss = loss_fn(y_proba, y)

            test_loss += loss.detach().cpu().item()
            test_accuracy += y_hat.eq(y).detach().cpu().mean(dtype=float)

            y = y.detach().cpu()[0]
            y_hat = y_hat.detach().cpu()[0]
            y_list.append(y)
            y_hat_list.append(y_hat)

            """
            if i < num_classes * 10:
                X = X.detach().cpu()[0]
                A = score[0].detach().cpu()[0]
                save_img(X, filename=f'img_{i % num_classes}_{i // num_classes}.csv', nrow=int(np.sqrt(cfg.dataset.bag_size)))
                save_score(A, filename=f'score_{i % num_classes}_{i // num_classes}.csv', nrow=int(np.sqrt(cfg.dataset.bag_size)))
                if cfg.model.name == 'additive':
                    P = score[1].detach().cpu()[0]
                    P = torch.transpose(P, 1, 0)
                    for j in range(num_classes):
                        save_score(P[j], filename=f'score_{i % num_classes}_{i // num_classes}_{j}.csv', nrow=int(np.sqrt(cfg.dataset.bag_size)))
            """
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)
    np.savetxt('test_loss.csv', [test_loss], delimiter=',')
    np.savetxt('test_accuracy.csv', [test_accuracy], delimiter=',')
    print('Test Loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_accuracy))

    y = np.array(y_list)
    y_hat = np.array(y_hat_list)
    np.savetxt('y_true.csv', y, delimiter=',')
    np.savetxt('y_pred.csv', y_hat, delimiter=',')

    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()