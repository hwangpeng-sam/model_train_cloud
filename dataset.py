import torch
from torch.utils.data import Dataset


class EvcDataset(Dataset):
    def __init__(self, rs, hs, ts, ss, ys):
        assert len(rs) == len(ys)

        self.rs = torch.tensor(rs).float()
        self.hs = torch.tensor(hs).float()
        self.ts = torch.tensor(ts).int()  # keep int dtype -> goes to embedding layer
        self.ss = torch.tensor(ss).float()
        self.ys = torch.tensor(ys).to(torch.int64)

    def __len__(self):
        return len(self.rs)

    def __getitem__(self, i):
        r, h, t, s, y = self.rs[i], self.hs[i], self.ts[i], self.ss[i], self.ys[i]
        return r, h, t, s, y
