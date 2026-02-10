# Training of recurrent neural networks for time-series tasks



**This repository contains the code of the paper**

Julian D. Schiller, Malte Heinrich, Victor G. Lopez, and Matthias A. Müller. Tuning the burn-in phase in training recurrent neural networks improves their performance. In: _Proceedings of the 14th International Conference on Learning Representations (ICLR)_, 2026, https://openreview.net/forum?id=jwkdKpioHJ.



The following experiments are provided:

1. Synthetic data experiment: small-scale system identification of academic nature (Section 5.1 of the paper)
2. Real-world data sets from the fields of system identification and time-series forecasting (Section 5.2 of the paper)

Training results described in the paper were achieved using the following software/hardware configurations:

- MATLAB, Version 25.2.0.2998904 (R2025b)

- CasADi, Version 3.7.2

- Python 3.12 (see requirements.txt)

- Ubuntu 24.04.3 LTS, Intel® Core™ Ultra 7 265H × 16, 32.0 GiB memory, NVIDIA RTX PRO 500 Blackwell GPU

  

## Synthetic data experiment

Train a one-dimensional linear RNN for a synthetic system identification task.

- Path: /synthetic_data_experiment
- Training script: main.m
- requirements: MATLAB, CasADi



## Public datasets: system identification and time-series forecasting

Train LSTMs for nonlinear system identification (SYSID) and time-series forecasting (TSF) using TBPTT.

- Path: /public_datasets

- Requirements:

  - Python (see requirements.txt)

  - MATLAB (for post-processing and visualization)

    

- Folders (/code):

  - /data: contains datasets for training/testing

  - /models: contains trained RNNs used in the paper

  - /results: contains the training results used in the paper

    

- Training scripts (/code):

  - ICLR_SYSID_LSTM_silverbox_WH.py (Silver-Box, Wiener-Hammerstein)

  - ICLR_SYSID_LSTM_RLC.py (RLC circuit)
  - ICLR_TSF_LSTM_electricity_traffic.py (Electricity, Traffic)
  - ICLR_TSF_LSTM_solar.py (Solar)
