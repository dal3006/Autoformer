import os

import numpy as np
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MyDataset
from model.autoformer import Autoformer
from utils.earlystoping import EarlyStopping
from utils.getdata import get_data


class EXP:
    def __init__(self, seq_len=96, label_len=48, pred_len=96, lr=0.0001, batch_size=32, epochs=10,
                 patience=3, verbose=True):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.lr = lr

        if not os.path.exists('./checkpoint/'):
            os.makedirs('./checkpoint/')

        self.modelpath = './checkpoint/autoformer_model.pkl'

        self._get_data()
        self._get_model()

    def _get_data(self):
        train, valid, test = get_data()

        trainset = MyDataset(train, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        validset = MyDataset(valid, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        testset = MyDataset(test, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)

        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        self.validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        if self.verbose:
            print('train: {0}, valid: {1}, test: {2}'.format(len(trainset), len(validset), len(testset)))

    def _get_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoformer(pred_len=self.pred_len).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))
        self.early_stopping = EarlyStopping(patience=self.patience, verbose=self.verbose, path=self.modelpath)
        self.criterion = nn.MSELoss()

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
        dec_inp = torch.cat([batch_x[:, -self.label_len:, :], dec_inp], dim=1).float().to(self.device)

        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.pred_len:, :]
        loss = self.criterion(outputs[:, -self.pred_len:, :], batch_y[:, -self.pred_len:, :])
        return outputs, loss

    def train(self):
        for e in range(self.epochs):
            self.model.train()
            train_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.trainloader):
                self.optimizer.zero_grad()
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            valid_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.validloader):
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                valid_loss.append(loss.item())

            test_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                test_loss.append(loss.item())

            train_loss, valid_loss, test_loss = np.average(train_loss), np.average(valid_loss), np.average(test_loss)
            print("Epoch: {0}, | Train Loss: {1:.4f} Vali Loss: {2:.4f} Test Loss: {3:.4f}".format(e + 1, train_loss,
                                                                                                   valid_loss,
                                                                                                   test_loss))

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break
            self.scheduler.step()
        self.model.load_state_dict(torch.load(self.modelpath))

    def test(self):
        self.model.eval()
        trues, preds = [], []
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
            pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.extend(pred.detach().cpu().numpy()[:, -self.pred_len:, :])
            trues.extend(batch_y.detach().cpu().numpy()[:, -self.pred_len:, :])

        trues, preds = np.array(trues), np.array(preds)
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        print('Test: MSE:{0:.4f}, MAE:{1:.6f}'.format(mse, mae))
