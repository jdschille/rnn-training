clc; clear; close all


print_tabs = false;
print_figs = false;
PATH_OUTPUT = '../../../paper/figures/';


% tsf data
dataset = 'electricity';
dataset = 'traffic';
dataset = 'solar';

dataset_name = ['results_' dataset '_h96.csv'];

data = readtable(dataset_name,'Delimiter',",");
N_list = unique(data.lookback_length_L);
N_list = N_list(1:end-1); % last one is BPTT
m_list = unique(data.m);

data_stateless_1 = data(strcmp(data.training_type, 'stateless'), :);
data_stateless_2 = data(strcmp(data.training_type, 'stateless retrain'), :);
data_stateful = data(strcmp(data.training_type, 'stateful'), :);
data_BPTT = data(strcmp(data.training_type, 'BPTT'), :);
%data_BPTT = data_BPTT(end,:);

data_1_N = cell(length(N_list),1);
data_2_N = cell(length(N_list),1);
data_sf_N = cell(length(N_list),1);
for i=1:length(N_list)
    data_1_N{i} = data_stateless_1(data_stateless_1.lookback_length_L == N_list(i), :);
    data_2_N{i} = data_stateless_2(data_stateless_2.lookback_length_L == N_list(i), :);
    data_sf_N{i} = data_stateful(data_stateful.lookback_length_L == N_list(i), :);
end

if strcmp(dataset,'electricity')
    y_limits = [0.07,0.235];
    select_stateful = 2;
    w = 0;
    m_star_min_perc = 0; % restrict m_star to % of m_max
    m_star_max = 1000;
    width_factor = 3;
    lgd_cols = 1;
    title_str = 'Electricity';
elseif strcmp(dataset,'traffic')
    y_limits = [0.18,0.4];
    select_stateful = 2;
    w = 0;
    m_star_min_perc = 0;
    m_star_max = 1000;
    width_factor = 2;
    lgd_cols = 1;
    title_str = 'Traffic';
elseif strcmp(dataset,'solar')
    y_limits = [0,0.38];
    select_stateful = 2;
    w = 0.2;
    m_star_min_perc = 0;
    m_star_max = 1000;
    width_factor = 2;
    lgd_cols = 1;
    title_str = 'Solar';
else
    y_limits = [0,1];
    select_stateful = 2;
    w=1;
    title_str = 'None';
    width_factor = 2;
    lgd_cols = 3;
end


% figure settings
height = 4.8;
width = 14.2;
fontsize = 9;
lw = 1;
ms = 4;
ms_small = ms;
gray = ones(1,3)*0.7;

c = colororder;
x_limits = [0,200];

f_all = create_fig(fontsize,height,width/width_factor);
hold on

% stateful
x_axis_static = linspace(x_limits(1),x_limits(end),5);
mse_train_sf = data_sf_N{select_stateful}.mse_train_scaled;
mse_test_sf = data_sf_N{select_stateful}.mse_val_scaled;
%plot(x_axis_static,ones(size(x_axis_static))*mse_train_sf,'-','Color',c(select_stateful,:),'LineWidth',lw,'MarkerSize',ms_small)
%plot(x_axis_static,ones(size(x_axis_static))*mse_test_sf,'--','Color',c(select_stateful,:),'LineWidth',lw,'MarkerSize',ms_small)

% bptt
res_bptt_train_mse = data_BPTT.mse_train_scaled;
res_bptt_test_mse = data_BPTT.mse_val_scaled;
plot(x_axis_static,ones(size(x_axis_static))*res_bptt_train_mse,'k-','Color',gray,'LineWidth',lw,'MarkerSize',ms_small);
plot(x_axis_static,ones(size(x_axis_static))*res_bptt_test_mse,'k--','Color',gray,'LineWidth',lw,'MarkerSize',ms_small)

m_star_idx_list = cell(size(N_list));
for i=1:length(N_list)
    x_axis = data_1_N{i}.m;%/max(data_1_N{i}.m);
    mse_1_train = data_1_N{i}.mse_train_scaled;
    mse_1_test = data_1_N{i}.mse_val_scaled;

    mse_1_mean = ((1-w)*mse_1_train+w*mse_1_test);
    mse_1_mean_min = min(mse_1_mean);
    mse_1_mean_min_idx = find(mse_1_mean==mse_1_mean_min);
    mse_1_mean_min_mval = x_axis(mse_1_mean_min_idx);
    while mse_1_mean_min_mval<m_star_min_perc/100*x_axis(end)
        mse_1_mean_min_idx = mse_1_mean_min_idx + 1;
        mse_1_mean_min_mval = x_axis(mse_1_mean_min_idx);
    end
    while mse_1_mean_min_mval>m_star_max
        mse_1_mean_min_idx = mse_1_mean_min_idx - 1;
        mse_1_mean_min_mval = x_axis(mse_1_mean_min_idx);
    end
    m_star_idx_list{i} = mse_1_mean_min_idx;

    mse_2_train = data_2_N{i}.mse_train_scaled;
    mse_2_test = data_2_N{i}.mse_val_scaled;

    %plot(x_axis,mse_1_train,'.','Color',c(i,:),'LineWidth',lw,'MarkerSize',ms)
    %plot(x_axis,mse_1_test,'+','Color',c(i,:),'LineWidth',lw,'MarkerSize',ms)


    plot(x_axis,mse_2_train,'o','Color',c(i,:),'LineWidth',lw,'MarkerSize',ms)
    plot(x_axis,mse_2_test,'x','Color',c(i,:),'LineWidth',lw,'MarkerSize',ms)

    % dummy plots for legend
    pts(i) = plot([-2,-1],[1e-3,1e-3],'-o','Color',c(i,:),'LineWidth',lw,'MarkerSize',ms);
end
hold off
%set(gca,'XScale','log')
%set(gca,'YScale','log')
xlim(x_limits)
ylim(y_limits)
%yticks(y_limits(1):0.02:y_limits(2))
title(title_str, 'Interpreter','tex','FontSize',fontsize,'FontName','Times')
xlabel('burn-in phase \it m', 'Interpreter','tex','FontSize',fontsize,'FontName','Times')
%ylabel('MSE', 'Interpreter','tex','FontSize',fontsize,'FontName','Times')
legend(pts,'\it N\rm =48','\it N\rm =96','\it N\rm =192',...
    'Interpreter','tex','FontSize',fontsize-1,'FontName','Times',...
    'NumColumns',lgd_cols, 'IconColumnWidth',15, ...
    'Location','northeast')
filestr = [PATH_OUTPUT 'TSF_' dataset '_N_m.pdf'];
if print_figs
    pause(1)
    exportgraphics(f_all,filestr)
end
%% compute metrics
rel_change = @(new,old) (new-old)/old*100;      

res_impr = table;
res_mse = table;
res_comp_time_avrg = inf(1,length(N_list)+1);

for i=1:length(N_list)
    idx = m_star_idx_list{i};
    res_m_star = data_2_N{i}.m(idx);
    res_mstar_train_mse = data_2_N{i}.mse_train_scaled(idx);
    res_mstar_test_mse = data_2_N{i}.mse_val_scaled(idx);
    res_m0_train_mse = data_2_N{i}.mse_train_scaled(end);
    res_m0_test_mse = data_2_N{i}.mse_val_scaled(end);
    res_sf_train_mse = data_sf_N{i}.mse_train_scaled(end);
    res_sf_test_mse = data_sf_N{i}.mse_val_scaled(end);
    res_impr_m0_train = rel_change(res_mstar_train_mse,res_m0_train_mse);
    res_impr_m0_test = rel_change(res_mstar_test_mse,res_m0_test_mse);
    res_impr_ss_train = rel_change(res_mstar_train_mse,res_sf_train_mse);
    res_impr_ss_test = rel_change(res_mstar_test_mse,res_sf_test_mse);
    res_impr_BPTT_train = rel_change(res_mstar_train_mse,res_bptt_train_mse);
    res_impr_BPTT_test = rel_change(res_mstar_test_mse,res_bptt_test_mse);

    res_impr(i,:) = {N_list(i),res_m_star, ...
        res_impr_m0_train, res_impr_ss_train, res_impr_BPTT_train,...
        res_impr_m0_test, res_impr_ss_test, res_impr_BPTT_test};

    res_mse(i,:) = {N_list(i),res_m_star, ...
        res_m0_train_mse, res_mstar_train_mse, res_sf_train_mse, res_bptt_train_mse,...
        res_m0_test_mse, res_mstar_test_mse, res_sf_test_mse, res_bptt_test_mse};


    res_comp_time_avrg(1,i) = mean([data_2_N{i}.time_training(:);data_sf_N{i}.time_training]);

end

res_comp_time_avrg(1,i+1) = data_BPTT.time_training;


res_impr.Properties.VariableNames = ["N","m_star",...
    "impr m0 train","impr ss train","impr bptt train",...
    "impr m0 test","impr ss test","impr bptt test"];

res_mse.Properties.VariableNames = ["N","m_star",...
    "m0 train","m_star train","ss train","bptt train",...
    "m0 test","m_star test","ss test","bptt test"];

res_impr

if print_tabs
    close all
    save(['workspace_results_' dataset '.mat'])
end
res_impr{:,3:end} = round(res_impr{:,3:end},0);

filename = ['table_' dataset '.csv'];
mse_scaling = 1;
res_mse{:,3:end} = round(res_mse{:,3:end}*mse_scaling,3);
if print_tabs
    writetable([res_mse,res_impr(:,[3,6])],filename,"Delimiter",",")
end

%% Helpers
function fig = create_fig(fontsize,height,width)

    fig = figure; 
    
    % Fonts
    set(fig,'DefaultAxesTickLabelInterpreter','tex')
    set(fig,'DefaultLegendInterpreter','tex')
    set(fig,'defaultTextInterpreter','tex')
    fig.Units = 'centimeters';
    box on
    grid off
    set(gca,'Color','none','FontName','Times','FontSize',fontsize)
    set(gcf, 'Color', 'w');
    
    % Dimensions
    plotsize = get(fig, 'Position');
    pos = [plotsize(1), plotsize(2), width, height];
    set(fig, 'Position', pos); 
end