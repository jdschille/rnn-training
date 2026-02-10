clear; clc; close all

print_figures = false;

% sysid dataset
%dataset = 'silverbox';
%dataset = 'wienerhammerstein';
%dataset = 'rlc';

% sysid dataname
%dataname = 'losses_N6667_m0_BPTT';
%dataname = 'losses_N500_m499_stateless';
%dataname = 'losses_N200_m199_stateless';
%dataname = 'losses_N100_m99_stateless';

% tsf dataset
%dataset = 'electricity';
%dataset = 'traffic';
%dataset = 'solar';

% dataset
dataset = 'electricity';
dataname_list = {...
    'id_2025_12_12-10_01_02',... % stateless retrain N48
    'id_2025_12_12-11_00_07',... % stateless retrain N96
    'id_2025_12_12-12_37_33',... % stateless retrain N192
    'id_2025_12_12-15_06_41',... % stateful N48
    'id_2025_12_12-15_12_27',... % stateful N96
    'id_2025_12_12-15_18_47',... % stateful N192
    'id_2025_12_12-15_26_03'};   % BPTT
mode = 'tsf';


% if strcmp(dataset,'electricity')
%     mode = 'tsf';
%     dataname_list = {...
%         'id_2025_12_03-07_04_35',... % stateless N48
%         'id_2025_12_03-07_25_23',... % stateless N96
%         'id_2025_12_03-07_59_52',... % stateless N192
%         'id_2025_12_03-08_59_06',... % stateless retrain N48
%         'id_2025_12_03-09_29_51',... % stateless retrain N96
%         'id_2025_12_03-10_20_59',... % stateless retrain N192
%         'id_2025_12_04-06_14_28',... % stateful N48
%         'id_2025_12_04-06_17_49',... % stateful N96
%         'id_2025_12_04-06_21_19',... % stateful N192
%         'id_2025_12_03-12_01_09'};   % BPTT
% 
% 
% elseif strcmp(dataset,'traffic')
%     mode = 'tsf';
%     dataname_list = {...
%         'id_2025_12_03-12_37_06',... % stateless N48
%         'id_2025_12_03-12_57_48',... % stateless N96
%         'id_2025_12_03-13_31_59',... % stateless N192
%         'id_2025_12_03-14_31_43',... % stateless retrain N48
%         'id_2025_12_03-15_03_35',... % stateless retrain N96
%         'id_2025_12_03-15_55_14',... % stateless retrain N192
%         'id_2025_12_04-06_25_53',... % stateful N48
%         'id_2025_12_04-06_29_11',... % stateful N96
%         'id_2025_12_04-06_33_14',... % stateful N192
%         'id_2025_12_03-17_36_36'};   % BPTT
% 
% elseif strcmp(dataset,'silverbox')
%     mode = 'sysid';
%     dataname_list = {...
%         'id_2025_12_04-06_47_22',... % stateful N100
%         'id_2025_12_04-06_49_01',... % stateful N200
%         'id_2025_12_04-06_50_49',... % stateful N500
%         'id_2025_12_04-06_52_58'};   % stateful N1000
% 
% elseif strcmp(dataset,'wienerhammerstein')
%     mode = 'sysid';
%     dataname_list = {...
%         'id_2025_12_02-01_12_53',... % stateful N100
%         'id_2025_12_02-01_45_25',... % stateful N200
%         'id_2025_12_02-02_35_58',... % stateful N500
%         'id_2025_12_02-03_56_37',...   % stateful N1000
%         'id_2025_12_04-06_56_19',... % stateful N100
%         'id_2025_12_04-06_58_13',... % stateful N200
%         'id_2025_12_04-07_00_08',... % stateful N500
%         'id_2025_12_04-07_02_24'};   % stateful N1000
% 
% else
% 
% end


if strcmp(mode,'sysid')
    title_list = {...
        'stateless N100',...
        'stateless N200',...
        'stateless N500',...
        'stateless N1000',...
        'stateful N100',...
        'stateful N200',...
        'stateful N500',...
        'stateful N1000',...
        'BPTT'};

else
    title_list = {...
        'stateless N48',...
        'stateless N96',...
        'stateless N192',...
        'stateless retrain N48',...
        'stateless retrain N96',...
        'stateless retrain N192',...
        'stateful N48',...
        'stateful N96',...
        'stateful N192',...
        'BPTT'};
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:length(dataname_list)

    dataname = dataname_list{i};
    data = readmatrix([dataset '/' dataname '.csv']);
    data(any(isnan(data),2),:)=[];
    
    window = 500;
    color_matrix = jet(size(data,2)-1);

    m_vals = 0:size(data,2)-2;

    m_lgd = cell(size(data,2)-1,1);
    for j =1:length(m_vals)
        m_lgd{j} = ['m_{' num2str(m_vals(j)) '}'];
    end


    fig = figure;
    ax = gca;
    ax.ColorOrder = color_matrix;
    hold on
    plot(data(:, 1),movmean(data(:, 2:end),window));
    hold off
    xlabel('time');
    ylabel('training loss');
    set(gca,"YScale","log");
    grid on;
    title_str = [dataset ' ' title_list{i}];
    title(title_str)
    if length(m_vals) > 1
        legend(m_lgd)
    end

    if print_figures
        exportgraphics(fig,[dataset '/' title_str,'.png'])
    end

end
