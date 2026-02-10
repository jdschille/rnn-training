# ------------------------------------------------------------
# Script for the training of an LSTM 
# benchmark example: RLC
# ------------------------------------------------------------

# --- IMPORTS --- #
import os
import time
from SYSID_functions import *
from LSTM_classdef import *
from datetime import datetime
from torch.utils.data import DataLoader

save_results = True

for train_mode in range(1,2):

    # --- DATASETS --- #
    path_data_train = "data/data_rlc_train_6667.csv"
    path_data_val = "data/data_rlc_val_3333.csv"
    path_data_test = "data/data_rlc_test_3333.csv"
    data_set_name = "rlc"

    # --- HYPERPARMETERS --- #
    hidden_size = 8
    learning_rate = 0.0003
    input_size = 1
    output_size = 2
    number_of_layers = 1

    if train_mode == 0:
        # --- TBPTT SETTINGS FOR MULTIPLE EXPERIMENTS ---
        training_type = "stateless"
        stateful = False
        timesteps_per_sequence_N_list = [100,200,500,1000]     # test for multiple N
        m_number_of_vals_list = [10,20,40,40]
        batch_size_b = 100
        num_of_epochs = 1500

    elif train_mode == 1:
        # override for stateful
        training_type = "stateful"
        stateful = True
        timesteps_per_sequence_N_list = [100,200,500,1000]      # test for multiple N, requires N mod b = 0
        m_number_of_vals_list = [1,1,1,1] #only m=0
        batch_size_b = 100
        num_of_epochs = 1500

    elif train_mode == 2:
        # override for BPTT
        training_type = "BPTT"
        stateful = False
        timesteps_per_sequence_N_list = [6667]
        m_number_of_vals_list = [1] #only m=0
        batch_size_b = 1
        num_of_epochs = 97500

    else:
        print('error')
        print(train_mode)

    # --- SCALING PARAMETERS --- #
    scaler = "StandardScaler" # select "StandardScaler" or "MinMaxScaler"

    # --- load training data --- #
    u_train, y_train, u_scaler, y_scaler, length_train = load_data_training(path_data_train, scaler, output_size)
    X_train_eval, _, _ = create_sequences_io_many_to_many_shift_by_one_mapping(u_train, y_train, length_train)

    # --- load validation data --- #
    u_val, y_val, length_val = load_data_testing(path_data_val, u_scaler, y_scaler, output_size)
    X_val, Y_val, _ = create_sequences_io_many_to_many_shift_by_one_mapping(u_val, y_val, length_val)

    # --- load testing data --- #
    u_test, y_test, length_test = load_data_testing(path_data_test, u_scaler, y_scaler, output_size)
    X_test, Y_test, _ = create_sequences_io_many_to_many_shift_by_one_mapping(u_test, y_test, length_test)

    # --- set cuda or mps as device if available for faster computations --- #
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    Nrun = 0
    for timesteps_per_sequence_N in timesteps_per_sequence_N_list:

        # --- MAIN --- #
        if __name__ == "__main__":


            # --- create dictionary for all losses and errors of different values of parameter m --- #
            loss_all = {}

            # --- create training batches --- #
            X_train, Y_train, indices_train = create_sequences_io_many_to_many_shift_by_one_mapping(u_train, y_train, timesteps_per_sequence_N)
            dataset = TensorDatasetIndexed(X_train, Y_train, indices_train)
            dataloader = DataLoader(dataset, batch_size=batch_size_b, shuffle = not stateful, drop_last=True)

            # --- select parameters m --- #
            m_number_of_vals = m_number_of_vals_list[Nrun]
            m_exceed = 20
            if m_number_of_vals > m_exceed:
                m_parameters_list = m_create_list(0, 99, m_exceed)
                m_parameters_list_exceed = m_create_list(110, timesteps_per_sequence_N - 1, m_number_of_vals-m_exceed)
                m_parameters_list += m_parameters_list_exceed
            else:
                m_parameters_list = m_create_list(0, timesteps_per_sequence_N-1, m_number_of_vals)

            # --- store current timestamp as id --- #
            id = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

            for m_loss_parameter in m_parameters_list:

                # --- set seed for same random initialization in all runs --- #
                set_seed()

                # --- create model --- #
                if training_type == "stateful":
                    model = LSTM_Stateful(input_size, hidden_size, number_of_layers, output_size,
                                                timesteps_per_sequence_N, batch_size_b, device).to(device)

                else:
                    model = LSTM_Stateless(input_size, hidden_size, number_of_layers, output_size,
                                                 device).to(device)

                # --- set loss function and optimizer --- #
                criterion = reduced_mse_loss
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                # --- list for all losses  of current parameter m --- #
                loss_list = []

                # --- perform training --- #
                start_time_train = time.time()
                if stateful:
                    train_model_stateful(model, dataloader, criterion, optimizer, num_of_epochs, m_loss_parameter,
                                          loss_list, device)
                else:
                    train_model_stateless(model, dataloader, criterion, optimizer, num_of_epochs, m_loss_parameter,
                                loss_list, device) #, cs_ih, cs_hh)
                end_time_train = time.time()

                if save_results:
                    # --- save trained model --- #
                    model_dir_name = 'models'
                    model_subdir_name = f'{data_set_name}'
                    model_file_name = f'model_N{timesteps_per_sequence_N}_m{m_loss_parameter}_{training_type}'
                    model_file_path = os.path.join(model_dir_name, model_subdir_name, model_file_name)
                    torch.save(model.state_dict(), model_file_path)

                # --- save losses for parameter m as value-key pair in a dictionary --- #
                loss_all[m_loss_parameter] = loss_list

                # --- evaluation of the training data --- #
                preds_scaled_train = evaluate_model(model, X_train_eval, device, manual_init=stateful)
                batch_size_train, seq_len_train, features_train = preds_scaled_train.shape
                preds_train = y_scaler.inverse_transform(preds_scaled_train.reshape(-1, features_train)).reshape(batch_size_train, seq_len_train, features_train)

                # --- evaluation of the validation data --- #
                preds_scaled_val = evaluate_model(model, X_val, device, manual_init=stateful)
                batch_size, seq_len, features = preds_scaled_val.shape
                preds_val= y_scaler.inverse_transform(preds_scaled_val.reshape(-1, features)).reshape(batch_size, seq_len, features)

                # --- evaluation of the testing data --- #
                preds_scaled_test = evaluate_model(model, X_test, device, manual_init=stateful)
                batch_size, seq_len, features = preds_scaled_test.shape
                preds_test = y_scaler.inverse_transform(preds_scaled_test.reshape(-1, features)).reshape(batch_size, seq_len, features)

                # --- get error metrics for predictions of training and testing data --- #
                (mse_train_scaled, mse_val_scaled, mse_test_scaled,
                        r2_train_scaled, r2_val_scaled, r2_test_scaled,
                        mse_train, mse_val, mse_test,
                        r2_train, r2_val, r2_test) = plot_train_evaluation_scores(output_size,
                                                    path_data_train, path_data_val, path_data_test,
                                                    preds_train=preds_train, preds_val=preds_val, preds_test=preds_test,
                                                    preds_train_scaled=preds_scaled_train, preds_val_scaled=preds_scaled_val, preds_test_scaled=preds_scaled_test,
                                                    y_train_scaled=y_train, y_val_scaled=y_val, y_test_scaled=y_test)

                # --- calculate robust value for the MSELoss due to fluctuations --- #
                mse_num_of_last_steps = 100
                mean_loss, std_loss = calculate_robust_mse_loss(loss_list, last_iterations=mse_num_of_last_steps)

                # --- save all current hyperparameters in a results dictionary --- #
                results = {
                    "timesteps_per_sequence_N": timesteps_per_sequence_N,
                    "m": m_loss_parameter,
                    "num_of_epochs": num_of_epochs,
                    "batch_size_b": batch_size_b,
                    "learning_rate": learning_rate,
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "number_of_layers": number_of_layers,
                    "output_size": output_size,

                    "r2_train_scaled": round(r2_train_scaled * 100, 2),
                    "r2_val_scaled": round(r2_val_scaled * 100, 2),
                    "r2_test_scaled": round(r2_test_scaled * 100, 2),
                    "mse_train_scaled": mse_train_scaled,
                    "mse_val_scaled": mse_val_scaled,
                    "mse_test_scaled": mse_test_scaled,

                    "r2_train": round(r2_train * 100, 2),
                    "r2_val": round(r2_val * 100, 2),
                    "r2_test": round(r2_test * 100, 2),
                    "mse_train": mse_train,
                    "mse_val": mse_val,
                    "mse_test": mse_test,

                    "mse_mean": mean_loss,
                    "mse_std": std_loss,

                    "time_training": round(end_time_train - start_time_train, 2),
                    "mse_num_of_last_steps": mse_num_of_last_steps,
                    "num_of_iterations": len(loss_list),
                    "id": id,
                    "weighted": "no",
                    "architecture": "LSTM",
                    "training_type": training_type,
                    "device": device.type,
                    "scaler": scaler,
                }

                if save_results:
                    # --- create .csv-file for saving the results --- #
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

            if save_results:
                # --- save losses to csv --- #
                losses_file_name = f'id_{id}'
                losses_dir_name = 'losses'
                losses_file_path = f"{losses_dir_name}/{data_set_name}/{losses_file_name}.csv"
                save_losses_in_csv(loss_all, losses_file_path)

        Nrun +=1