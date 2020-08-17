import torch

from corr_decorr_python.tensors import one_hot


class LSTM(torch.nn.Module):
    def __init__(self, element_params, hidden_units, dropout):
        super().__init__()
        self.vocab_size = element_params
        self.hidden_units = hidden_units
        lhu = self.vocab_size
        for i, hu in enumerate(self.hidden_units):
            setattr(
                self,
                f'lstm{i}',
                torch.nn.LSTM(lhu, hu, 1, batch_first=True, dropout=dropout)
            )
            lhu = hu
        self.output_linear = torch.nn.Linear(lhu, self.vocab_size)
        self.softmax = torch.nn.LogSoftmax(-1)

    def get_lstms(self):
        lstms = []
        i = 0
        while hasattr(self, f'lstm{i}'):
            lstms.append(getattr(self, f'lstm{i}'))
            i += 1
        return lstms

    def forward(self, x):
        x = one_hot(x, self.vocab_size, 'cpu:0')
        lstms = self.get_lstms()
        for lstm in lstms:
            x, _ = lstm(x)
        logits = self.output_linear(x)
        return self.softmax(logits)
