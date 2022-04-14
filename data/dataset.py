from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, seq_len=96, label_len=48, pred_len=96):
        self.data = data[0]
        self.stamp = data[1]

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        e_begin = index
        e_end = e_begin + self.seq_len
        d_begin = e_end - self.label_len
        d_end = e_end + self.pred_len

        seq_x = self.data[e_begin:e_end]
        seq_y = self.data[d_begin:d_end]
        seq_x_mark = self.stamp[e_begin:e_end]
        seq_y_mark = self.stamp[d_begin:d_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
