# -----------------------------------------------------------------
# Check trained LSTMs for compliance with Assumption 1
# Here: system identification (Silver-Box, Wiener-Hammerstein, RLC)
# -----------------------------------------------------------------

from SYSID_functions import *
import torch.nn as nn
import numpy as np


# --- Parameters --- #

# dataset
dataset_name = 'silverbox'
dataset_name = 'wienerhammerstein'
dataset_name = 'rlc'

# model to check
model_name = "model_N500_m499_stateless"
model_name = "model_N100_m99_stateless"

# bound parameters
C = 50
lam = 0.9999

# number of random initial conditions (h0,c0) and std to check
n_sample = 1000

# each (h0,c0) is sampled from normal distribution with zero mean and standard deviation std_hc
std_hc = 0.1

# number of input sequences to evaluate for each initial condition (randomly sampled from the training data)
batch_size = 200


# load data set-specific parameters
if dataset_name == 'silverbox':
    data_path = "data/data_silverbox_train_20000.csv"
    output_size = 1
elif dataset_name == 'wienerhammerstein':
    data_path = "data/data_WienerHammerBenchmark_train_20000.csv"
    output_size = 1
elif dataset_name == 'rlc':
    data_path = "data/data_rlc_train_6667.csv"
    output_size = 2
else:
    raise ValueError('Dataset unknown')

model_dir = f'models/{dataset_name}'


def create_sequences(x):

    X, indices = [], []

    for i in range(len(x)):
        X.append(np.concatenate((x[i:],np.zeros((i,np.size(x,axis=1)),dtype=np.float32))))
        indices.append(i)
    return np.array(X), np.array(indices)

u_train, y_train, u_scaler, y_scaler, length_train = load_data_training(data_path, scaler = "MinMaxScaler", output_size=output_size)
seq_len = len(u_train)
dataset_X, dataset_indices = create_sequences(u_train)

def create_bound(C, lam, seq_len):

    b = np.ones(shape=(seq_len, 1))

    for i in range(seq_len):
        b[i] = C * lam ** i

    return b

b_t = create_bound(C, lam, seq_len)

# LSTM setup
hidden_size = 8
input_size = np.size(u_train,axis=1)
number_of_layers = 1

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, number_of_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

device = torch.device("cuda")
model = LSTM()
model_path = f'{model_dir}/{model_name}'
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# test condition
for sample in range(n_sample):

    if sample % 100 == 0:
        print(f'run {sample} of {n_sample}')

    # randomly pick batch of input data
    batch_indices = np.random.choice(dataset_indices, size=batch_size, replace=False)
    batch_X = torch.tensor(dataset_X[batch_indices, :])

    # randomly sample initial cell states
    s1 = torch.normal(mean=0,std=std_hc,size=(number_of_layers, batch_size, 2 * hidden_size))
    s2 = torch.normal(mean=0,std=std_hc,size=(number_of_layers, batch_size, 2 * hidden_size))
    h1 = s1[:, :, 0:hidden_size]
    h2 = s2[:, :, 0:hidden_size]
    c1 = s1[:, :, hidden_size:]
    c2 = s2[:, :, hidden_size:]
    dh = np.linalg.norm(h1 - h2, axis=2)

    # evaluate model
    with torch.no_grad():
        y1 = model(batch_X.to(device), h1.to(device), c1.to(device))
        y2 = model(batch_X.to(device), h2.to(device), c2.to(device))

    y1 = y1.cpu().numpy()
    y2 = y2.cpu().numpy()
    dy = np.linalg.norm(y1 - y2, axis=2)

    # compare outputs with bound
    bound = np.outer(b_t, dh).T
    diff = dy-bound
    cond = diff > 0
    bound_violated = np.asarray(cond).nonzero()

    count_viol = 0
    viol_batches = []
    for batch_idx in bound_violated[0]:
        if batch_idx not in viol_batches:
            print(f'violated in run {sample}, batch {batch_idx}, starting at time {bound_violated[1][count_viol]}')
            dy_b = dy[batch_idx,:]
            bound_b = bound[batch_idx, :]
            res_viol = np.vstack((bound_b,dy_b))
            count_viol += 1
            viol_batches.append(batch_idx)