# -------------------------------------------------------------------------- #
# Script with functions for the training of RNNs for time series forecasting #
# -------------------------------------------------------------------------- #


# --- IMPORTS --- #
import csv
import torch
import random
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn


# --- FUNCTION DEFINITIONS --- #

def create_seq2seq_mapping(x, lookback_length_L, forecasting_length_H):
    """ This function creates artificial io sequences X and Y for time series with a many-to-many mapping and shift-by-one """

    X, Y, indices_X, indices_Y = [], [], [], []
    
    for i in range(len(x) - lookback_length_L - forecasting_length_H+1):
        X.append(x[i:i+lookback_length_L])
        Y.append([x[i+j+1:i+j+1+forecasting_length_H] for j in range(lookback_length_L)])
        indices_X.append(i)
        indices_Y.append([i+j+1 for j in range(lookback_length_L)])

    return np.array(X), np.array(Y)[:,:,:,0], np.array(indices_X), np.array(indices_Y)


def create_val_data(z, lookback_length_L, forecasting_length_H):
    """ This function creates X Y data for validation """

    X = z[:-forecasting_length_H]
    Y = z[lookback_length_L:]

    return np.array(X), np.array(Y)


def get_data_length(path):
    """ This function returns the total number of points of a time series """

    df = pd.read_csv(path, header=0)

    data_length = len(df.columns)

    return data_length


def load_data_training(path, length_train, variate, scaler):
    """ This function loads training data from a .csv-file and performs a Scaling """

    df = pd.read_csv(path, header=0)

    z = np.array(df.iloc[variate, 0:length_train], dtype=np.float32).reshape(-1, 1)

    if scaler == "StandardScaler":
        z_scaler = StandardScaler()
    elif scaler == "MinMaxScaler":
        z_scaler = MinMaxScaler()

    z_scaled = z_scaler.fit_transform(z)

    return z_scaled, z_scaler, z


def load_data_training_unscaled(path, length_train, variate):
    """ This function loads training data from a .csv-file without data scaling """

    df = pd.read_csv(path, header=0)

    z = np.array(df.iloc[variate, 0:length_train], dtype=np.float32).reshape(-1, 1)

    return z


def load_data_validation(path, z_scaler, length_train, length_val, lookback_length_L, variate):
    """ This function loads validation data from a .csv-file """

    df = pd.read_csv(path, header=0)

    z = np.array(df.iloc[variate, length_train-lookback_length_L:length_train+length_val], dtype=np.float32).reshape(-1, 1)

    z_scaled = z_scaler.transform(z)

    return z_scaled, z


def load_data_validation_unscaled(path, length_train, length_val, lookback_length_L, variate):
    """ This function loads validation data from a .csv-file without data scaling """

    df = pd.read_csv(path, header=0)

    z = np.array(df.iloc[variate, length_train-lookback_length_L:length_train+length_val], dtype=np.float32).reshape(-1, 1)

    return z


def load_data_testing(path, z_scaler, length_train, length_val, length_test,  lookback_length_L, variate):
    """ This function loads testing data from a .csv-file """

    df = pd.read_csv(path, header=0)

    z = np.array(df.iloc[variate, length_train+length_val-lookback_length_L:length_train+length_val+length_test], dtype=np.float32).reshape(-1, 1)

    z_scaled = z_scaler.transform(z)

    return z_scaled, z


def train_model_stateless(model, dataloader, criterion, optimizer, num_of_epochs, m_loss_parameter,
                loss_list, device):
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

            for xb, yb, _, _  in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                output = model(xb)
                loss = criterion(output, yb, m_loss_parameter)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                loss_list.append(loss.item())

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

            for xb, yb, _, _ in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                output = model(xb)
                loss = criterion(output, yb, m_loss_parameter)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

            progress.advance(task)


def evaluate_model(model, dataloader, device, manual_init = False):
    """ This function performs the evaluation for the fully trained model """

    # manually reset hidden states for stateful mode
    if manual_init:
        model.reset_hidden_state()

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            sequences, _, _, _ = batch
            sequences = sequences.to(device)

            output = model(sequences)

            for i in range(output.size(0)):
                predictions.append(output[i].cpu())

    return torch.stack(predictions).numpy()  

def evaluate_model_seq(model, x_seq, device, manual_init = False):
    """ This function performs the evaluation for the fully trained model """

    # --- manually reset hidden states for stateful mode --- #
    if manual_init:
        model.reset_hidden_state()

    model.eval()

    with torch.no_grad():
        x = torch.tensor(x_seq.squeeze()).to(device)
        x = x[None,:,None]
        output = model(x)
        predictions = output[0,-1,:].detach().cpu().numpy().reshape(-1, 1)

    return predictions


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


def train_evaluation_scores(dataloader_train_scaled, dataloader_train_unscaled,
                                preds_scaled_train=None, preds_unscaled_train=None):
    """ This function returns error metrics """

    # --- get all predicted forecasting trajectories for the last step in lookback sequence --- #
    preds_train_last_unscaled = preds_unscaled_train[:, -1, :]
    preds_train_last_scaled = preds_scaled_train[:, -1, :]

    # --- turn dataloader into tensors (unscaled values) --- #
    dataloader_train_tensor_unscaled = dataloader_to_output_tensor(dataloader_train_unscaled)
    true_train_unscaled = dataloader_train_tensor_unscaled[:, -1, :].detach().cpu().numpy()

    # --- turn dataloader into tensors (scaled values) --- #
    dataloader_train_tensor_scaled = dataloader_to_output_tensor(dataloader_train_scaled)
    true_train_scaled = dataloader_train_tensor_scaled[:, -1, :].detach().cpu().numpy()

    # --- calculate scores --- #
    r2_train_unscaled = r2_score(true_train_unscaled.squeeze(), preds_train_last_unscaled.squeeze())
    mse_train_unscaled = mean_squared_error(true_train_unscaled, preds_train_last_unscaled)
    r2_train_scaled = r2_score(true_train_scaled.squeeze(), preds_train_last_scaled.squeeze())
    mse_train_scaled = mean_squared_error(true_train_scaled, preds_train_last_scaled)

    return r2_train_unscaled, mse_train_unscaled, r2_train_scaled, mse_train_scaled


def val_evaluation_scores(y_true_scaled, y_true_unscaled, y_pred_scaled, y_pred_unscaled):
    """ This function returns error metrics """

    # --- calculate scores --- #
    r2_train_unscaled = r2_score(y_true_unscaled, y_pred_unscaled)
    mse_train_unscaled = mean_squared_error(y_true_unscaled, y_pred_unscaled)
    r2_train_scaled = r2_score(y_true_scaled, y_pred_scaled)
    mse_train_scaled = mean_squared_error(y_true_scaled, y_pred_scaled)

    return r2_train_unscaled, mse_train_unscaled, r2_train_scaled, mse_train_scaled

def dataloader_to_output_tensor(dataloader, device=None):
    """ This function turns all input batches of a data loader in one stacked tensor """
    outputs = []

    for batch in dataloader:
        yb = batch[1]
        if device is not None:
            yb = yb.to(device)

        for i in range(yb.size(0)):
            outputs.append(yb[i].cpu())

    return torch.stack(outputs)


def save_losses_in_csv(losses, filename):
    """ This function saves all losses for different values of parameter m into a .csv file"""

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
    step = (stop - start) / float(count - 1) if count > 1 else 1
    return [round(start + i * step) for i in range(count)]

# >>>>> CLASS DEFINITIONS >>>>> #

class TensorDatasetIndexed(torch.utils.data.Dataset):
    """ This class creates a Pytorch dataset with index tracking """

    def __init__(self, X, Y, indices_X, indices_Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.indices_X = torch.tensor(indices_X, dtype=torch.long)
        self.indices_Y = torch.tensor(indices_Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.indices_X[index], self.indices_Y[index]
