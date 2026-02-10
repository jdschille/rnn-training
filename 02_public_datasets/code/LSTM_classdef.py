# ------------------------------------------------------ #
# LSTM class definitions for stateful and stateless mode #
# ------------------------------------------------------ #

# --- IMPORTS --- #
import torch
import torch.nn as nn

# --- CLASS: Definition of stateless LSTM model (hidden state is set to 0 for every batch) --- #
class LSTM_Stateless(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, number_of_layers: int, output_size: int, device):

        super(LSTM_Stateless, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.number_of_layers = number_of_layers
        self.output_size = output_size
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, number_of_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.number_of_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.number_of_layers, x.size(0), self.hidden_size).to(self.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

# --- CLASS: Definition of stateful LSTM model (hidden state is passed between batches) --- #
class LSTM_Stateful(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, number_of_layers: int, output_size: int,
                 sequence_length: int, batch_size: int, device):

        # check if N mod b = 0
        assert sequence_length % batch_size == 0

        super(LSTM_Stateful, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.number_of_layers = number_of_layers
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.number_of_hidden_state_slots = self.sequence_length // self.batch_size
        self.hidden_state_slots = [None] * self.number_of_hidden_state_slots
        self.batch_count = 0          # gets increased in every forward pass

        self.lstm = nn.LSTM(input_size, hidden_size, number_of_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def init_hidden_state(self, batch_size):
        h0 = torch.zeros(self.number_of_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.number_of_layers, batch_size, self.hidden_size).to(self.device)
        return (h0, c0)

    def reset_hidden_state(self):
        self.hidden_state_slots = [None] * self.number_of_hidden_state_slots
        self.batch_count = 0

    def forward(self, x):
        slot_num = self.batch_count % self.number_of_hidden_state_slots

        hidden_state = self.hidden_state_slots[slot_num]
        if hidden_state is None or hidden_state[0].size(1) != x.size(0):
            hidden_state = self.init_hidden_state(x.size(0))

        out, hidden_state = self.lstm(x, hidden_state)
        out = self.fc(out)

        self.hidden_state_slots[slot_num] = (hidden_state[0].detach(), hidden_state[1].detach())

        self.batch_count += 1
        return out