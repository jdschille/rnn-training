# --------------------------------------------------------------------- #
# Script with functions for the training of RNNs for SYSID applications #
# --------------------------------------------------------------------- #


# --- IMPORTS --- #
import csv
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn


# --- FUNCTION DEFINITIONS --- #

def create_sequences_io_many_to_one_mapping(x, y, length):
    """ This function creates sequences for the inputs x and y with a many-to-one mapping """

    X, Y = [], []
    
    for i in range(len(x) - length):
        X.append(x[i:i+length])
        Y.append(y[i+length])

    return np.array(X), np.array(Y)


def create_sequences_io_many_to_many_shift_by_one_mapping(x, y, length):
    """ This function creates sequences for the inputs x and y with a many-to-many mapping and shift-by-one """

    X, Y, indices = [], [], []
    
    for i in range(len(x) - length + 1):
        X.append(x[i:i+length])
        Y.append(y[i:i+length])
        indices.append(i)

    return np.array(X), np.array(Y), np.array(indices)


def create_sequences_io_many_to_many_shift_by_length_mapping(x, y, length):
    """ This function creates sequences for the inputs x and y with a many-to-many mapping and shift-by-length """

    X, Y = [], []
    num_of_slices = len(x) // length

    for i in range(0, num_of_slices * length, length):
        X.append(x[i : i + length])
        Y.append(y[i : i + length])

    return np.array(X), np.array(Y)


def load_data_training(path, scaler, output_size = 1):
    """ This function loads training data from a .csv-file and performs a MinMaxScaling """

    df = pd.read_csv(path, header=None)

    u = np.array(df.iloc[-output_size-1, 0:], dtype=np.float32).reshape(-1, 1)
    y = np.flip(np.array(df.iloc[-output_size:, 0:], dtype=np.float32).T, axis=1)

    if scaler == "StandardScaler":
        u_scaler = StandardScaler()
        y_scaler = StandardScaler()
    elif scaler == "MinMaxScaler":
        u_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()

    u_scaled = u_scaler.fit_transform(u)
    y_scaled = y_scaler.fit_transform(y)

    length_train = len(u)

    return u_scaled, y_scaled, u_scaler, y_scaler, length_train


def load_data_testing(path, u_scaler, y_scaler, output_size = 1):
    """ This function loads testing data from a .csv-file """

    df = pd.read_csv(path, header=None)

    u = np.array(df.iloc[-output_size-1, 0:], dtype=np.float32).reshape(-1, 1)
    y = np.flip(np.array(df.iloc[-output_size:, 0:], dtype=np.float32).T, axis=1)

    u_scaled = u_scaler.transform(u)
    y_scaled = y_scaler.transform(y)

    length_test = len(u)

    return u_scaled, y_scaled, length_test

def project_rowwise_max(weights, c):
    """ This function performs row-wise clamping of a weight matrix """

    with torch.no_grad():
        return torch.clamp(weights, min=-c, max=c)


def project_rowwise_scaling(weights, cs, hidden_size):
    """ This function performs row-wise scaling of a weight matrix """

    with torch.no_grad():

        idx = {
            'i': slice(0 * hidden_size, 1 * hidden_size),
            'f': slice(1 * hidden_size, 2 * hidden_size),
            'g': slice(2 * hidden_size, 3 * hidden_size),
            'o': slice(3 * hidden_size, 4 * hidden_size),
        }

        weights_norm = weights.clone()

        for gate in ['i', 'f', 'g', 'o']:
            block = weights[idx[gate], :]
            c = cs[gate]

            row_max = block.abs().max(dim=1, keepdim=True).values
            scale = torch.clamp(c / row_max, max=1.0)

            weights_norm[idx[gate], :] = block * scale

        return weights_norm

def train_model_stateless(model, dataloader, criterion, optimizer, num_of_epochs, m_loss_parameter,
                 loss_list, device): #, None=cs_w_ih, cs_w_hh):
    """ This function performs the training loop of a stateless LSTM model """

    model.train()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Training..."),
        BarColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Training", total=num_of_epochs)

        for epoch in range(num_of_epochs):
            total_loss = 0

            for xb, yb, indices in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                output = model(xb)
                loss = criterion(output, yb, m_loss_parameter)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                loss_list.append(loss.item())

                # project weight matrices
                hidden_size = model.lstm.hidden_size
                #model.lstm.weight_ih_l0.data = project_rowwise_scaling(model.lstm.weight_ih_l0.data, cs_w_ih, hidden_size)
                #model.lstm.weight_hh_l0.data = project_rowwise_scaling(model.lstm.weight_hh_l0.data, cs_w_hh, hidden_size)


            progress.advance(task)


def train_model_stateful(model, dataloader, criterion, optimizer, num_of_epochs, m_loss_parameter,
                         loss_list, device):
    """ This function performs the training loop of a stateful LSTM model """

    model.train()

    with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Training..."),
            BarColumn(),
            TimeRemainingColumn(),
            transient=True,
    ) as progress:
        task = progress.add_task("Training", total=num_of_epochs)

        for epoch in range(num_of_epochs):
            model.reset_hidden_state()

            for xb, yb, indices in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                output = model(xb)
                loss = criterion(output, yb, m_loss_parameter)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

            progress.advance(task)


def evaluate_model(model, dataset, device, manual_init=False):
    """ This function performs the evaluation of the trained model on the specified data set """

    # --- manually reset hidden states for stateful mode --- #
    if manual_init:
        model.reset_hidden_state()

    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(dataset, dtype=torch.float32).to(device))

    return preds.cpu().numpy()


def reduced_mse_loss(predictions, targets, m_loss_parameter):
    """ This function calculates the reduced MSE loss truncated until m-th summation """

    predictions = predictions[:, m_loss_parameter:, :]
    targets = targets[:, m_loss_parameter:, :]
    
    error_squared = (predictions - targets) ** 2
    red_mse_err = torch.mean(error_squared)
    
    return red_mse_err


def set_seed(seed=0):
    """ This function sets a seed to get comparable results when random initializations are used """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_robust_mse_loss(loss_list, last_iterations=10):
    """ This function determines a robust value of the MSELoss as the arithmetic mean """

    if len(loss_list) < last_iterations:
        last_iterations = len(loss_list)
    
    last_loss_values = loss_list[-last_iterations:]
    mean_loss = np.mean(last_loss_values)
    std_loss = np.std(last_loss_values)

    return mean_loss, std_loss


def cut_trajectories(preds):
    """ This function cuts all the trajectories for model evaluation to only last value except for first sequence (there all samples) """

    plot_x = []
    plot_y = []

    for i, trajectory in enumerate(preds):
        #start = indices[i] if indices is not None else 0
        if i == 0:
            x_vals = list(range(i, i + len(trajectory)))
            plot_x.extend(x_vals)
            plot_y.extend(trajectory)
        else:
            plot_x.append(i + len(trajectory) - 1)
            plot_y.append(trajectory[-1])
    
    return plot_x, plot_y


def plot_train_evaluation_scores(output_size, path_data_train, path_data_val, path_data_test,
                                preds_train=None, preds_val=None, preds_test=None,
                                preds_train_scaled=None, preds_val_scaled=None, preds_test_scaled=None,
                                y_train_scaled=None, y_val_scaled=None, y_test_scaled=None):
    """ This function returns error metrics """

    # --- store testing and training data in pandas dataframe --- #
    df_train = pd.read_csv(path_data_train, header=None)
    df_val = pd.read_csv(path_data_val, header=None)
    df_test = pd.read_csv(path_data_test, header=None)

    # --- get ground truth data for all training inputs and outputs --- #
    y_train = np.flip(np.array(df_train.iloc[-output_size:, 0:], dtype=np.float32).T, axis=1)
    y_val = np.flip(np.array(df_val.iloc[-output_size:, 0:], dtype=np.float32).T, axis=1)
    y_test = np.flip(np.array(df_test.iloc[-output_size:, 0:], dtype=np.float32).T, axis=1)

    # --- squeeze predictions --- #
    preds_train = preds_train[0, :, :]
    preds_val = preds_val[0, :, :]
    preds_test = preds_test[0, :, :]
    preds_train_scaled = preds_train_scaled[0, :, :]
    preds_val_scaled = preds_val_scaled[0, :, :]
    preds_test_scaled = preds_test_scaled[0, :, :]

    # --- calculate r2 scores --- #
    r2_train = r2_score(preds_train, y_train)
    r2_val = r2_score(preds_val, y_val)
    r2_test = r2_score(preds_test, y_test)
    r2_train_scaled = r2_score(preds_train_scaled, y_train_scaled)
    r2_val_scaled = r2_score(preds_val_scaled, y_val_scaled)
    r2_test_scaled = r2_score(preds_test_scaled, y_test_scaled)

    # --- calculate mse loss --- #
    mse_train = mean_squared_error(preds_train, y_train)
    mse_val = mean_squared_error(preds_val, y_val)
    mse_test = mean_squared_error(preds_test, y_test)
    mse_train_scaled = mean_squared_error(preds_train_scaled, y_train_scaled)
    mse_val_scaled = mean_squared_error(preds_val_scaled, y_val_scaled)
    mse_test_scaled = mean_squared_error(preds_test_scaled, y_test_scaled)

    return (mse_train_scaled, mse_val_scaled, mse_test_scaled,
            r2_train_scaled, r2_val_scaled, r2_test_scaled,
            mse_train, mse_val, mse_test,
            r2_train, r2_val, r2_test)


def save_losses_in_csv(losses, filename):
    """ This function saves all losses for different values of parameter m into a .csv file """

    max_len = max(len(v) for v in losses.values())

    m_values = sorted(losses.keys())

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        header = ["iteration"] + [f"{m}" for m in m_values]
        writer.writerow(header)

        for i in range(max_len):
            row = [i]
            for m in m_values:

                if i < len(losses[m]):
                    row.append(losses[m][i])
                else:
                    row.append("")
            writer.writerow(row)

def m_create_list(start, stop, count):
    """ This function creates a list for values of parameter m """
    step = (stop - start) / float(count-1) if count > 1 else 1
    return [round(start + i * step) for i in range(count)]



# >>>>> CLASS DEFINITIONS >>>>> #

class TensorDatasetIndexed(torch.utils.data.Dataset):
    """ This class creates a Pytorch dataset with index tracking """

    def __init__(self, X, Y, indices):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.indices = torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.indices[index]