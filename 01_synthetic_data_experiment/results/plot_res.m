clear; close all; clc

load('tests.mat') % load data

print_figs_1 = false;
print_figs_2 = false;
PATH_OUTPUT = '';


height_y_avrg = 3;
height_app = 10;
width = 14.2;
fontsize = 9;
lw = 0.8;
ms = 7;
lw_ms = 0.9;
lw_y = 2.5;
m_select = [1,6]; % show y solutions for specific m
cl.white = [1,1,1];
cl.grey = [1,1,1]*0.65;
cl.lgrey = [1,1,1]*0.94;
clo = colororder;


%% batch-averaged output error e_j
fig_yavrg = create_fig(fontsize,height_y_avrg,0.53*width);
lgd_entries = cell(size(m_set));
hold on
for m_idx = 1:length(m_set)
    plot(1:N,Y_AVRG{m_idx}.S,'LineWidth',lw)
    lgd_entries{m_idx} = ['$m = ' num2str(m_set(m_idx)) '$'];
end
hold off
xlim([1,N])
xticks(linspace(1,N,5))
ylim([1e-9,4e-1])
yticks(logspace(-9,-1,5))
grid on
set(gca,'YScale','log')
xlabel('time $j$', 'FontName', 'Times','FontSize',fontsize)
ylabel('avrg. output error $e_j$', 'FontName', 'Times','FontSize',fontsize)
lgd = legend(lgd_entries,'Location','southwest','NumColumns',2,'IconColumnWidth',9,'Box','on');
lgd.Position = [0.17,0.238520153942584,0.399872374104922,0.290596019574349];
filestr = [PATH_OUTPUT 'ex_academic_y_avrg.pdf'];
if print_figs_1
    pause(1)
    exportgraphics(fig_yavrg,filestr)
end

%% performance (MSE) for TRAINING & TEST
figMSE = create_fig(fontsize,height_y_avrg,0.45*width);
hold on
p_train = plot(m_set,RES_MSE_train(1:2:length(m_set)*2,4),'-','Color',cl.grey,'LineWidth',lw);
plot(m_set,RES_MSE_train(2:2:length(m_set)*2,4),'-','Color',cl.grey,'LineWidth',lw);
p_test = plot(m_set,RES_MSE_test(1:2:length(m_set)*2,3),'-.','Color',cl.grey,'LineWidth',lw);
plot(m_set,RES_MSE_test(2:2:length(m_set)*2,3),'-.','Color',cl.grey,'LineWidth',lw);
for m_idx = 1:length(m_set)
    plot(m_set(m_idx),RES_MSE_train((m_idx)*2-1,4),'o','Color',clo(m_idx,:),'MarkerSize',ms,'LineWidth',lw_ms);
    plot(m_set(m_idx),RES_MSE_train((m_idx)*2,4),'x','Color',clo(m_idx,:),'MarkerSize',ms,'LineWidth',lw_ms);
end
for m_idx = 1:length(m_set)
    plot(m_set(m_idx),RES_MSE_test((m_idx)*2-1,3),'o','Color',clo(m_idx,:),'MarkerSize',ms,'LineWidth',lw_ms);
    plot(m_set(m_idx),RES_MSE_test((m_idx)*2,3),'x','Color',clo(m_idx,:),'MarkerSize',ms,'LineWidth',lw_ms);
end
hold off
set(gca,'YScale','log')
ylim([1e-4,4e-2])
yticks(logspace(-4,-1,4))
xticks(m_set)
grid on
xlabel('burn-in phase length $m$','FontName', 'Times','FontSize',fontsize)
ylabel('performance $P$','FontName', 'Times','FontSize',fontsize)
lgd = legend([p_test,p_train],'test data','training data',...
    'FontName','Times','FontSize',fontsize,'IconColumnWidth',20);
lgd.Position = [0.434418493605597,0.392439001674341,0.439532447530359,0.214838705370503];
filestr = [PATH_OUTPUT 'ex_academic_T_y.pdf'];
if print_figs_1
    pause(1)
    exportgraphics(figMSE,filestr)
end

%% Appendix
fig_app = create_fig(fontsize,height_app,width);
tiledlayout(3,2,'TileSpacing','Compact','Padding','compact');

% data
nexttile
hold on
plot(data_train.t,data_train.X,'k','LineWidth',lw)
plot(data_train.t,data_train.Y,'Color',cl.grey,'LineWidth',lw)
hold off
set(gca,'Layer', 'top')
box on
grid on
xlim([1,T])
lgd = legend('$x_t^\mathrm{d}$','$y^\mathrm{d}_t$',...
    'IconColumnWidth',9,'Location','northeast');

nexttile
hold on
plot(data_test.t,data_test.X,'Color','k','LineWidth',lw)
plot(data_test.t,data_test.Y,'Color',cl.grey,'LineWidth',lw)
hold off
set(gca,'Layer', 'top')
box on
grid on
xlim([1,T])

% test predictions
tile = 3;
for m_idx = m_select
    nexttile(tile)
    hold on
    for j = 1:S
        p0t = plot(RNN_0{m_idx}.SI(j):RNN_0{m_idx}.SI(j)+N-1,RNN_0{m_idx}.Y{j}.Y,'Color',clo(m_idx,:),'LineWidth',lw);
    end
    pd = plot(data_train.t,data_train.Y,'Color',cl.grey,'LineWidth',lw_y);
    p1 = plot(data_train.t,RNN_1_T_train{m_idx}.Y,'r-','LineWidth',lw);
    p0 = plot(data_train.t,RNN_0_T_train{m_idx}.Y,'b-.','LineWidth',lw);
        

    lgd = legend([pd,p0t,p0,p1],'$y^\mathrm{d}_t$',...
        '${y}^*_{j|i}$','${y}^*_t$','$y^\mathrm{b}_t$',...
        'Location','northeast','IconColumnWidth',11);
    hold off
    box on
    grid on
    xlim([1,T])
    ylim([-1.02,1.02])
    tile = 5;
end
xlim([1,T])
xticks([1,20:20:T])
xlabel('time $t$', 'FontName', 'Times','FontSize',fontsize)


% test predictions
tile = 4;
for m_idx = m_select
    nexttile
    hold on
    pd = plot(data_test.t,data_test.Y,'Color',cl.grey,'LineWidth',lw_y);
    p1 = plot(data_test.t,RNN_1_T_test{m_idx}.Y,'r-','LineWidth',lw);
    p0 = plot(data_test.t,RNN_0_T_test{m_idx}.Y,'b-.','LineWidth',lw);
    hold off
    box on
    ylim([-1,1])
    xlim([1,T])
    xticks([1,20:20:T])
    grid on
    tile = 6;
end
xlabel('time $t$', 'FontName', 'Times','FontSize',fontsize)

% print figure
filestr = [PATH_OUTPUT 'ex_academic_app.pdf'];
if print_figs_2
    pause(1)
    exportgraphics(fig_app,filestr)
end



%% Helpers
function fig = create_fig(fontsize,height,width)

    fig = figure; 
    
    % Fonts
    set(fig,'DefaultAxesTickLabelInterpreter','latex')
    set(fig,'DefaultLegendInterpreter','latex')
    set(fig,'defaultTextInterpreter','latex')
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