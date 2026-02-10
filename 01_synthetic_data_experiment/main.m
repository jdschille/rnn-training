% RNN training for dynamical systems

% Code for the simulation example in Section 5.1 in the paper:
% Schiller, Heinrich, Lopez & Müller, "Tuning the burn-in phase in training
% recurrent neural networks improves their performance", ICLR 2026.

clear; close all
yalmip('clear')
import casadi.*

%% Settings

save_and_show_res = true;
save_str = 'results/tests';

% General
rand_setup.fix_seed = true; % set true for using the same warmstart
rand_setup.seed = 11111;

% Data segmentation
T = 100; % length full training sequence
N = 21; % length subsequence for TBPTT
S = T-N+1; % number of subsequences for training
m_set = round(linspace(0,N-1,6)); % simulate for multiple burn-in phases
Ttest = T; % length test sequence

% RNN setup
RNN.n_input = 1;
RNN.n_hidden = 1;
RNN.n_output = 1;
RNN.activation_h = @(x) x;
RNN.activation_y = @(x) x;
RNN.f = @(h,x,W_hh,W_xh,b_h) RNN.activation_h(W_hh*h + W_xh*x + 0*b_h);
RNN.g = @(h,W_hy,b_y) RNN.activation_y(W_hy*h + 0*b_y);
RNN.n_learnables = RNN.n_hidden^2+(RNN.n_input+RNN.n_output+1)*RNN.n_hidden+RNN.n_output;

%% data generation LTI
sys = tf([5 1],[-2 1 0],1);

data_train.t = 1:T;
data_train.X = sin(1e-1*data_train.t) + sin(5e-2*data_train.t+2);
data_train.Y = lsim(sys,data_train.X,data_train.t)';
noise_train = 2e-1;
data_train.Y = data_train.Y + (-noise_train+2*noise_train*rand(size(data_train.Y)));

data_test.t = 1:Ttest;
data_test.X = -1.4*cos(1e-1*data_test.t) + 0.5*cos(0.5*data_test.t+1);
data_test.Y = lsim(sys,data_test.X,data_test.t)';
noise_test = 1e-0;
data_test.Y = data_test.Y + (-noise_test+2*noise_test*rand(size(data_test.Y)));

normalize_data = true;
if normalize_data
    scale_fun = @(x) 2*(x-min(x))/(max(x)-min(x))-1;
    data_train.X = scale_fun(data_train.X);
    data_train.Y = scale_fun(data_train.Y);
    data_test.X = scale_fun(data_test.X);
    data_test.Y = scale_fun(data_test.Y);
end

%% RNN training
RNN_0=cell(1,length(m_set)); % TBPTT RNN (fixed initial conditions)
RNN_1=cell(1,length(m_set)); % benchmark RNN (coupled initial conditions)
RNN_3=cell(1,length(m_set)); % produces y^\infty (free initial conditions)

Y_AVRG=cell(1,length(m_set)); % averaged output prediction error

for m_idx=1:length(m_set)
    custom_warmstart = []; % only for case 0
    delay_idx = 1; % only for case 1; delay_idx=N-overlay_i
    RNN_0{m_idx} = RNN_.learn(T,N,S,m_set(m_idx),RNN,data_train,0,rand_setup,custom_warmstart,delay_idx);
    RNN_1{m_idx} = RNN_.learn(T,N,S,m_set(m_idx),RNN,data_train,1,rand_setup,custom_warmstart,delay_idx);

    custom_warmstart.H = RNN_1{m_idx}.H;
    custom_warmstart.theta = RNN_1{m_idx}.theta;
    RNN_3{m_idx} = RNN_.learn(T,N,S,m_set(m_idx),RNN,data_train,3,rand_setup,custom_warmstart,delay_idx);

    % compute avrg. output prediction error
    Y_sum = zeros(1,N);
    for j = 1:S
        Y_sum = Y_sum + vecnorm(RNN_0{m_idx}.Y{j}.Y-RNN_1{m_idx}.Y{j}.Y,2,1).^2;
    end
    Y_AVRG{m_idx}.S = Y_sum./S;

end

%% check coercivity
% verify the "additional technical condition" of Thm1/2
% -> that is,satisfaction of the property in equation (25) for epsilon>0
% -> throws warning if not satisfied
epsilon = 2;
RNN_0 = check_coercivity(RNN_0,RNN_3,m_set,data_train,epsilon);
RNN_1 = check_coercivity(RNN_1,RNN_3,m_set,data_train,epsilon);

%% prediction training data
RNN_0_T_train=cell(1,length(m_set));
RNN_1_T_train=cell(1,length(m_set));
RNN_3_T_train=cell(1,length(m_set));
for m_idx=1:length(m_set)
    t1 = 1;
    t2 = m_set(m_idx)+1;
    t3 = T;
    RNN_0_T_train{m_idx} = RNN_.predict(RNN_0{m_idx},RNN_0{m_idx}.H{1}.H(:,1),data_train,t1,t2,t3);
    RNN_1_T_train{m_idx} = RNN_.predict(RNN_1{m_idx},RNN_1{m_idx}.H{1}.H(:,1),data_train,t1,t2,t3);
    RNN_3_T_train{m_idx} = RNN_.predict(RNN_3{m_idx},RNN_3{m_idx}.H{1}.H(:,1),data_train,t1,t2,t3);
end

RES_MSE_train = inf(length(m_set)*2,5);
j=1;
for m_idx=1:length(m_set)
    RES_MSE_train(j,:) = [m_set(m_idx), RNN_0{m_idx}.obj,RNN_0_T_train{m_idx}.MSE_Y_t1t2,RNN_0_T_train{m_idx}.MSE_Y_t2t3,RNN_0_T_train{m_idx}.MSE_Y_t1t3];
    RES_MSE_train(j+1,:) = [m_set(m_idx),RNN_1{m_idx}.obj,RNN_1_T_train{m_idx}.MSE_Y_t1t2,RNN_1_T_train{m_idx}.MSE_Y_t2t3,RNN_1_T_train{m_idx}.MSE_Y_t1t3];
    j=j+2;
end

%% prediction test data
RNN_0_T_test=cell(1,length(m_set));
RNN_1_T_test=cell(1,length(m_set));
for m_idx=1:length(m_set)
    t1 = 1;
    t2 = m_set(m_idx)+1;
    t3 = Ttest;
    RNN_0_T_test{m_idx} = RNN_.predict(RNN_0{m_idx},RNN_0{m_idx}.H{end}.H(:,end)*0,data_test,t1,t2,t3);
    RNN_1_T_test{m_idx} = RNN_.predict(RNN_1{m_idx},RNN_1{m_idx}.H{end}.H(:,end)*0,data_test,t1,t2,t3);
end

RES_MSE_test = inf(length(m_set)*2,4);
j=1;
for m_idx=1:length(m_set)
    RES_MSE_test(j,:) = [m_set(m_idx),RNN_0_T_test{m_idx}.MSE_Y_t1t2,RNN_0_T_test{m_idx}.MSE_Y_t2t3,RNN_0_T_test{m_idx}.MSE_Y_t1t3];
    RES_MSE_test(j+1,:) = [m_set(m_idx),RNN_1_T_test{m_idx}.MSE_Y_t1t2,RNN_1_T_test{m_idx}.MSE_Y_t2t3,RNN_1_T_test{m_idx}.MSE_Y_t1t3];
    j=j+2;
end

%% save and plot data
if save_and_show_res
    close all
    save(save_str)

    addpath('results')
    plot_res
    rmpath('results')
end