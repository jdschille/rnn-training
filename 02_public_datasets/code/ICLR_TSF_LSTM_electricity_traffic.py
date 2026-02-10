# ------------------------------------------------------------
# Script for the training of an LSTM 
# Benchmark example: Electricity, Traffic
# ------------------------------------------------------------

# --- IMPORTS --- #
import os
import time
from datetime import datetime
from TSF_functions import *
from LSTM_classdef import *
from torch.utils.data import DataLoader


save_results = True

for dataset_mode in range(0,2):

    # --- DATASETS --- #
    if dataset_mode == 0:
        path_data = "data/data_electricity_0_10_hours_kWh.csv"
        data_set_name = "electricity"
        variate = 5

    else:
        path_data = "data/data_traffic.csv"
        data_set_name = "traffic"
        variate = 9


    # --- Common parameters --- #
    scaler = "StandardScaler"

    # --- Common hyperparameters --- #
    forecasting_length_H = 96
    input_size = 1
    hidden_size = 96
    output_size = input_size * forecasting_length_H
    number_of_layers = 1
    learning_rate = 0.00005

    for train_mode in range(2,3):

        # set sequence lengths
        length_train = 6640  # training samples
        length_val = 3360  # test/validation samples

        if train_mode == 0:
            # --- TBPTT train ---
            training_type = "stateless"
            stateful = False
            lookback_length_L_list = [48, 96, 192]
            m_number_of_vals_list = [10, 15, 20]
            batch_size_b = 24
            num_of_epochs = 400
            retrain = False

        elif train_mode == 1:
            # retrain on validation set
            training_type = "stateless retrain"
            stateful = False
            lookback_length_L_list = [48, 96, 192]
            m_number_of_vals_list = [10, 15, 20]
            batch_size_b = 24
            num_of_epochs = 400
            retrain = True


        elif train_mode == 2:
            # stateful training
            training_type = "stateful"
            stateful = True
            lookback_length_L_list = [48, 96, 192]
            m_number_of_vals_list = [1, 1, 1] # only m=L-1
            batch_size_b = 24
            num_of_epochs = 400
            retrain = False

        else:
            # BPTT
            training_type = "BPTT"
            stateful = False
            lookback_length_L_list = [length_train + length_val - forecasting_length_H]
            m_number_of_vals_list = [1] # only m=L-1
            batch_size_b = 1
            num_of_epochs = 141600
            retrain = False

            # --- FOR RETRAINING + TESTING TBPTT (AND BPTT) ---
            # 1.1 train lstm on training data (length_train)
            # 1.2 validate/select hyperparameters (m) using validation data (length_val)
            # 2.1 re-train with selected hyperparameters (set length_train = length_train + length_val)
            # 2.2 evaluate test performance on new data (length_val)

        # --- adjust length for re-training, stateful, BPTT
        if (training_type in ["BPTT","stateful"]) or retrain:
            length_train = length_train + length_val

        # --- set cuda or mps as device if available for faster computations --- #
        device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        Lrun = 0
        for lookback_length_L in lookback_length_L_list:

            # --- MAIN --- #
            if __name__ == "__main__":

                # --- create dictionary for all losses and errors of different values of parameter m --- #
                loss_all = {}

                # --- load training data from .csv-file and create sequences --- #
                z_train, z_scaler, z_train_unscaled = load_data_training(path_data, length_train, variate, scaler)
                X_train, Y_train, indices_train_X, indices_train_Y = create_seq2seq_mapping(z_train, lookback_length_L, forecasting_length_H)

                # --- load training data from .csv-file and create sequences (unscaled!) --- #
                z_train_eval_unscaled = load_data_training_unscaled(path_data, length_train, variate)
                X_train_eval_unscaled, Y_train_eval_unscaled, indices_train_X_eval_unscaled, indices_train_Y_eval_unscaled = create_seq2seq_mapping(z_train_eval_unscaled, lookback_length_L, forecasting_length_H)
                dataset_train_eval_unscaled = TensorDatasetIndexed(X_train_eval_unscaled, Y_train_eval_unscaled, indices_train_X_eval_unscaled, indices_train_Y_eval_unscaled)
                dataloader_train_eval_unscaled = DataLoader(dataset_train_eval_unscaled, batch_size=batch_size_b, shuffle=False, drop_last=False)

                # --- load validation data from .csv-file and create sequences --- #
                z_val, z_val_unscaled = load_data_validation(path_data, z_scaler, length_train, length_val, lookback_length_L, variate)
                X_val, Y_val = create_val_data(z_val, lookback_length_L, forecasting_length_H)

                # --- load validation data from .csv-file and create sequences (unscaled!) --- #
                z_val_eval_unscaled = load_data_validation_unscaled(path_data, length_train, length_val, lookback_length_L, variate)
                X_val_unscaled, Y_val_unscaled = create_val_data(z_val, lookback_length_L, forecasting_length_H)

                # --- select parameters m --- #
                m_number_of_vals = m_number_of_vals_list[Lrun]
                m_parameters_list = m_create_list(0, lookback_length_L-1, m_number_of_vals)
                if stateful:
                    m_parameters_list = [lookback_length_L-1]

                # --- store current timestamp as id --- #
                id = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

                for m_loss_parameter in m_parameters_list:

                    # --- set seed for same random initialization in all runs --- #
                    set_seed()

                    # --- create pytorch tensors and batches with index tracking --- #
                    dataset_train = TensorDatasetIndexed(X_train, Y_train, indices_train_X, indices_train_Y)
                    dataloader_train = DataLoader(dataset_train, batch_size=batch_size_b, shuffle = not stateful, drop_last=True)
                    dataloader_train_eval = DataLoader(dataset_train, batch_size=batch_size_b, shuffle=False, drop_last=False)

                    # --- create model --- #
                    if training_type == "stateful":
                        model = LSTM_Stateful(input_size, hidden_size, number_of_layers, output_size,
                                                    lookback_length_L, batch_size_b, device).to(device)

                    else:
                        model = LSTM_Stateless(input_size, hidden_size, number_of_layers, output_size,
                                                     device).to(device)


                    # --- set loss function and optimizer --- #
                    criterion = reduced_mse_loss
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                    # --- list for all losses and reduced_error_sum_norms of current parameter m --- #
                    loss_list = []

                    # --- perform training --- #
                    start_time_train = time.time()
                    if stateful:
                        train_model_stateful(model, dataloader_train, criterion, optimizer, num_of_epochs, m_loss_parameter,
                                                loss_list, device)
                    else:
                        train_model_stateless(model, dataloader_train, criterion, optimizer, num_of_epochs, m_loss_parameter,
                                                loss_list, device)
                    end_time_train = time.time()


                    # --- save trained model --- #
                    if save_results:
                        model_dir_name = 'models'
                        model_subdir_name = f'{data_set_name}'
                        model_file_name = f'model_N{lookback_length_L}_m{m_loss_parameter}_{training_type}'
                        model_file_path = os.path.join(model_dir_name, model_subdir_name, model_file_name)
                        torch.save(model.state_dict(), model_file_path)

                    # --- save losses and reduced_error_sum_norms for parameter m as value-key pair in a dictionary --- #
                    loss_all[m_loss_parameter] = loss_list

                    # --- evaluation of the training data --- #
                    preds_scaled_train = evaluate_model(model, dataloader_train_eval, device, manual_init=stateful)
                    batch_size_train, seq_len_train, forecasting_length_train = preds_scaled_train.shape
                    preds_train = z_scaler.inverse_transform(preds_scaled_train.reshape(-1, 1)).reshape(batch_size_train, seq_len_train, forecasting_length_train)

                    print("finished training ... started evaluation")

                    # --- get error metrics for predictions of training data --- #
                    (r2_train_unscaled, mse_train_unscaled,
                     r2_train_scaled, mse_train_scaled) = train_evaluation_scores(
                                                    dataloader_train_scaled=dataloader_train_eval,
                                                    dataloader_train_unscaled=dataloader_train_eval_unscaled,
                                                    preds_scaled_train=preds_scaled_train,
                                                    preds_unscaled_train=preds_train)

                    # --- evaluation of the validation data --- #
                    res_tuple = (0, 0, 0, 0) # (r2_val_unscaled, mse_val_unscaled, r2_val_scaled, mse_val_scaled)
                    for i in range(length_val - forecasting_length_H + 1):

                        x_seq = X_val[i:i+lookback_length_L]
                        preds_val_scaled = evaluate_model_seq(model, x_seq, device, manual_init=stateful)
                        preds_val = z_scaler.inverse_transform(preds_val_scaled)

                        y_true_scaled = Y_val[i:i+forecasting_length_H]
                        y_true_unscaled = Y_val_unscaled[i:i+forecasting_length_H]

                        res_tuple_i = val_evaluation_scores(y_true_scaled, y_true_unscaled, preds_val_scaled, preds_val)
                        res_tuple = tuple(x + y for x, y in zip(res_tuple, res_tuple_i))

                    res_tuple = tuple(t / length_val for t in res_tuple)

                    r2_val_unscaled, mse_val_unscaled, r2_val_scaled, mse_val_scaled = res_tuple

                    # --- calculate robust value for the loss due to fluctuations --- #
                    mse_num_of_last_steps = 1000
                    mean_loss, std_loss = calculate_robust_mse_loss(loss_list, last_iterations=mse_num_of_last_steps)

                    # --- save all current hyperparameters in a results dictionary --- #
                    results = {
                        "lookback_length_L": lookback_length_L,
                        "m": m_loss_parameter,
                        "forecasting_length_H": forecasting_length_H,
                        "input_size": input_size,
                        "hidden_size": hidden_size,
                        "number_of_layers": number_of_layers,
                        "output_size": output_size,
                        "num_of_epochs": num_of_epochs,
                        "batch_size_b": batch_size_b,
                        "learning_rate": learning_rate,
                        "length_train": length_train,
                        "length_val": length_val,
                        "variate": variate,

                        "mse_train_scaled": mse_train_scaled,
                        "mse_val_scaled": mse_val_scaled,
                        "r2_train_scaled": round(r2_train_scaled * 100, 2),
                        "r2_val_scaled": round(r2_val_scaled * 100, 2),

                        "mse_train_unscaled": mse_train_unscaled,
                        "mse_val_unscaled": mse_val_unscaled,
                        "r2_train_unscaled": round(r2_train_unscaled * 100, 2),
                        "r2_val_unscaled": round(r2_val_unscaled * 100, 2),

                        "loss_mean_last1000": mean_loss,
                        "loss_std_last1000": std_loss,

                        "time_training": round(end_time_train - start_time_train, 2),
                        "num_of_iterations": len(loss_list),
                        "id": id,
                        "weighted": "no",
                        "architecture": "LSTM",
                        "training_type": training_type,
                        "device": device.type,
                        "scaler": scaler,
                    }

                    # --- create .csv-file for saving the results --- #
                    if save_results:
                        csv_dir_name = 'results'
                        csv_file_name = f'{csv_dir_name}_{data_set_name}_h{hidden_size}.csv'
                        csv_path_name = os.path.join(csv_dir_name, csv_file_name)
                        os.makedirs(csv_dir_name, exist_ok=True)
                        write_head_column = not os.path.exists(csv_path_name)

                        # --- write the current results to .csv-file --- #
                        with open(csv_path_name, mode="a", newline="") as file:
                            writer = csv.DictWriter(file, fieldnames=results.keys())
                            if write_head_column:
                                writer.writeheader()
                            writer.writerow(results)

                    # --- print the current results to console --- #
                    print(str(results))

                # --- save losses to csv --- #
                if save_results:
                    losses_file_name = f'id_{id}'
                    losses_dir_name = 'losses'
                    losses_file_path = f"{losses_dir_name}/{data_set_name}/{losses_file_name}.csv"
                    save_losses_in_csv(loss_all, losses_file_path)

            Lrun += 1