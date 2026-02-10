clc; clear; close all

print_figs = false;
print_tabs = false;

PATH_OUTPUT = '../../../paper/figures/';


dataset = 'silverbox';
dataset = 'wienerhammerstein';
dataset = 'rlc';

% load data
data = readtable(['results_' dataset '_h8.csv']);
N_list = unique(data.timesteps_per_sequence_N);
N_list = N_list(1:end-1);
m_list = unique(data.m);

data_stateless = data(strcmp(data.training_type, 'stateless'), :);
data_stateful = data(strcmp(data.training_type, 'stateful'), :);
data_BPTT = data(strcmp(data.training_type, 'BPTT'), :);

data_stateless_N = cell(length(N_list),1);
data_stateful_N = cell(length(N_list),1);
for i=1:length(N_list)
    data_stateless_N{i} = data_stateless(data_stateless.timesteps_per_sequence_N == N_list(i), :);
    data_stateful_N{i} = data_stateful(data_stateful.timesteps_per_sequence_N == N_list(i), :);
end

%% compute metrics

if strcmp(dataset,'silverbox')
    w = 0.3;
    m_star_max = 95;
    y_limits = [1.7e-5,1e-2];
    title_str = 'Silver-Box';
    width_factor = 2.7;
elseif strcmp(dataset,'wienerhammerstein')
    w = 0.3;
    m_star_max = 1000;
    y_limits = [2.5e-4,6e-2];
    title_str = 'Wiener-Hammerstein';
    width_factor = 2;
elseif strcmp(dataset,'rlc')
    w = 0.3;
    m_star_max = 1000;
    y_limits = [2.5e-4,0.4e-1];
    title_str = 'RLC';
    width_factor = 2;
else
    w = 0.3;
    m_star_max = 1000;
    fix_m_to_max = false;
    y_limits = [2e-5,1.2e-2];

end

% find m_star
for i=1:length(N_list)

    m_vals = data_stateless_N{i}.m;
    mse_1_train = data_stateless_N{i}.mse_train_scaled;
    mse_1_val = data_stateless_N{i}.mse_val_scaled;

    mse_1_mean = ((1-w)*mse_1_train+w*mse_1_val);
    mse_1_mean_min = min(mse_1_mean);
    mse_1_mean_min_idx = find(mse_1_mean==mse_1_mean_min);
    mse_1_mean_min_mval = m_vals(mse_1_mean_min_idx);
    m_star_idx_list{i} = mse_1_mean_min_idx;

end

res_impr = table;
res_mse = table;
res_comp_time_avrg = inf(1,length(N_list)+1);

res_bptt_train_mse = data_BPTT.mse_train_scaled;
res_bptt_test_mse = data_BPTT.mse_test_scaled;

rel_change = @(new,old) (new-old)/old*100;      
for i=1:length(N_list)

    mse_train = data_stateless_N{i}.mse_train_scaled;
    mse_val = data_stateless_N{i}.mse_val_scaled;
    mse_test = data_stateless_N{i}.mse_test_scaled;

    mse_train_sf = data_stateful_N{i}.mse_train_scaled;
    mse_test_sf = data_stateful_N{i}.mse_test_scaled;
        

    % get m_star ()constrained to 100
    idx = m_star_idx_list{i};
    res_m_star = data_stateless_N{i}.m(idx);
    while res_m_star > m_star_max
        idx = idx - 1;
        res_m_star = data_stateless_N{i}.m(idx);
    end

    res_mstar_train_mse = mse_train(idx);
    res_mstar_test_mse = mse_test(idx);
    res_m0_train_mse = mse_train(1);
    res_m0_test_mse = mse_test(1);
    res_sf_train_mse = mse_train_sf;
    res_sf_test_mse = mse_test_sf;
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


    res_comp_time_avrg(1,i) = mean([data_stateless_N{i}.time_training(:);data_stateful_N{i}.time_training]);
end

res_comp_time_avrg(1,i+1) = data_BPTT.time_training;

res_impr

res_impr.Properties.VariableNames = ["N","m_star",...
    "impr m0 train","impr ss train","impr bptt train",...
    "impr m0 test","impr ss test","impr bptt test"];

res_mse.Properties.VariableNames = ["N","m_star",...
    "m0 train","m_star train","ss train","bptt train",...
    "m0 test","m_star test","ss test","bptt test"];

if print_tabs
    save(['workspace_results_' dataset '.mat'])
end
res_impr{:,3:end} = round(res_impr{:,3:end},0);


mse_scaling = 100;
res_mse{:,3:end} = round(res_mse{:,3:end}*mse_scaling,3);
filename = ['table_' dataset '.csv'];
if print_tabs
    %res_impr(:,1:2) = [];
    writetable([res_mse,res_impr(:,[3,6])],filename,"Delimiter",",")
end

% %% plot single instance
% select_i = 3;
% 
% f_single = create_fig(fontsize,height,width/2);
% c = colororder;
% set(gca,'XScale','log')
% set(gca,'YScale','log')
% x_limits = [4.5,1000];
% y_limits = [2e-5,1.2e-2];
% hold on
% for i=1:length(N_list)
%     x_axis = data_stateless_N{i}.m;%/max(data_stateless_N{i}.m);
%     x_axis(1) = 1e-3;
%     mse_train = data_stateless_N{i}.mse_train_scaled;
%     mse_test = data_stateless_N{i}.mse_test_scaled;
% 
%     mse_train_sf = data_stateful_N{i}.mse_train_scaled;
%     mse_test_sf = data_stateful_N{i}.mse_test_scaled;
% 
%     if ismember(i,select_i)
%         plot(x_axis,mse_train,'o','Color',c(i,:),'LineWidth',lw,'MarkerSize',ms)
%         plot(x_axis,mse_test,'x','Color',c(i,:),'LineWidth',lw,'MarkerSize',ms)
%         plot([x_limits(1),x_limits(end)],[mse_train_sf,mse_train_sf],'-','Color',c(i,:),'LineWidth',lw,'MarkerSize',ms)
%         plot([x_limits(1),x_limits(end)],[mse_test_sf,mse_test_sf],'--','Color',c(i,:),'LineWidth',lw,'MarkerSize',ms)
%     end
% end
% 
% plot([x_limits(1),x_limits(end)],[mse_train_bptt,mse_train_bptt],'-','Color',gray,'LineWidth',lw,'MarkerSize',ms)
% plot([x_limits(1),x_limits(end)],[res_bptt_test,res_bptt_test],'--','Color',gray,'LineWidth',lw,'MarkerSize',ms)
% hold off
% xlim(x_limits)
% ylim(y_limits)


%% plot all

% figure settings
height = 4.8;
width = 14.2;
fontsize = 9;
lw = 1;
ms = 4;
gray = ones(1,3)*0.7;

c = colororder;
x_limits = [2,1000];

f_all = create_fig(fontsize,height,width/width_factor);
hold on

% stateful
select_stateful = 1;
mse_train_sf = data_stateful_N{select_stateful}.mse_train_scaled;
mse_test_sf = data_stateful_N{select_stateful}.mse_test_scaled;
%plot([x_limits(1),x_limits(end)],[mse_train_sf,mse_train_sf],'-','Color',c(select_stateful,:),'LineWidth',lw)
%plot([x_limits(1),x_limits(end)],[mse_test_sf,mse_test_sf],'--','Color',c(select_stateful,:),'LineWidth',lw)

% bptt
plot([x_limits(1),x_limits(end)],[res_bptt_train_mse,res_bptt_train_mse],'k-','Color',gray,'LineWidth',lw);
plot([x_limits(1),x_limits(end)],[res_bptt_test_mse,res_bptt_test_mse],'k--','Color',gray,'LineWidth',lw)

for i=1:length(N_list)

    x_axis = data_stateless_N{i}.m;%/max(data_stateless_N{i}.m);
    x_axis(1) = x_limits(1);
    mse_train = data_stateless_N{i}.mse_train_scaled;
    mse_test = data_stateless_N{i}.mse_test_scaled;
    plot(x_axis,mse_train,'o','Color',c(i,:),'LineWidth',lw,'MarkerSize',ms)
    plot(x_axis,mse_test,'x','Color',c(i,:),'LineWidth',lw,'MarkerSize',ms)

    % validation data
    %mse_val = data_stateless_N{i}.mse_val_scaled;
    %plot(x_axis,mse_val,'+','Color',c(i,:),'LineWidth',lw,'MarkerSize',ms)

    % dummy plots for legend
    pts(i) = plot([1e-4,1e-3],[1e-3,1e-3],'-o','Color',c(i,:),'LineWidth',lw,'MarkerSize',ms);

end

hold off
set(gca,'XScale','log')
set(gca,'YScale','log')
xlim(x_limits)
ylim(y_limits)
title(title_str, 'Interpreter','tex','FontSize',fontsize,'FontName','Times')
xlabel('burn-in phase \it m', 'Interpreter','tex','FontSize',fontsize,'FontName','Times')
%ylabel('MSE', 'Interpreter','tex','FontSize',fontsize,'FontName','Times')
legend(pts,'\it N\rm =100','\it N\rm =200','\it N\rm =500','\it N\rm =1000',...
    'Interpreter','tex','FontSize',fontsize-1,'FontName','Times',...
    'NumColumns',2, 'IconColumnWidth',15, ...
    'Location','south')
filestr = [PATH_OUTPUT 'SYSID_' dataset '_N_m.pdf'];
if print_figs
    pause(1)
    exportgraphics(f_all,filestr)
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