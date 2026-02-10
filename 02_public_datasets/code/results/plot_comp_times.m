close all; clear; clc


print_figs = true;


workspace_data = [...
    "workspace_results_silverbox.mat",...
    "workspace_results_wienerhammerstein.mat",...
    "workspace_results_rlc.mat",...
    "workspace_results_electricity.mat",...
    "workspace_results_traffic.mat",...
    "workspace_results_solar.mat"];



% figure settings
height = 4.8;
width = 14.2;
fontsize = 9;
c = colororder;
gray = ones(1,3)*0.7;



f_ct = create_fig(fontsize,height,width/4);
hold on
for k=1:length(workspace_data)
    load(workspace_data(k))
    cscale = res_comp_time_avrg(end);
    ctimes = res_comp_time_avrg/cscale;
    Nvals = N_list;

    c_bptt = ctimes(end);
    c_tbptt = ctimes(1:end-1);
    wscale = 0.6;
    bar(k,c_bptt,wscale,'FaceColor',gray,'LineStyle','none');
    for j = length(c_tbptt):-1:1
        if j==1, wscale = wscale *0.5; end
        bar(k,ctimes(j),wscale,'FaceColor',c(j,:),'LineStyle','none');
    end
    ct_bptt_str = [num2str(round(cscale)) 's'];
    if k ==3
        clr = [1,1,1];
    else
        clr = [0,0,0];
    end
    text(k,1,ct_bptt_str,'Rotation',90,'HorizontalAlignment','right','FontSize',fontsize,'Color',clr,'FontName','Times')
end
hold off
ax = gca;
set(ax,'Yscale','log')
grid on
set(ax, 'XMinorGrid', 'off', 'YMinorGrid', 'on')
xticks(1:length(workspace_data))
xspace = 0.7;
xlim([1-xspace,length(workspace_data)+xspace])
ylim([1e-2,1])
ax.XTickLabels = {'Silver-Box','W-H','RLC','Electricity','Traffic','Solar'};
ax.XTickLabelRotation = 90;

PATH_OUTPUT = '../../../paper/figures/';
filestr = [PATH_OUTPUT 'ex_comptimes.pdf'];
if print_figs
    pause(2)
    exportgraphics(f_ct,filestr)
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