%% A demo code to save ROC curve and bar chart for evaluating salient object ranking algorithms
% please cite our paper "Revisiting Salient Object Detection:
% Simultaneous Detection, Ranking, and Subitizing of Multiple Salient Objects",
% CVPR 2018, if you use the code in your research
% This code was originally modified from code provided for: "Contextual Hypergraph Modeling for Salient Object
% Detection", ICCV 2013

clear all; close all;

dataset = 'PASCAL-S'; % name of the dataset

%% Methods to draw
%allmethods = {'DSR','HS','MC','MDF','MTDS','DHS','NLDF','DSS','AMULET','UCF','RSDNET-R'}; % you can add more names of methods separated by comma
%methods =  {'DSR','HS','MC','MDF','MTDS','DHS','NLDF','DSS','AMULET','UCF','RSDNET-R'};

allmethods = {'RSDNET'}; % you can add more names of methods separated by comma
methods =  {'RSDNET'};

methods_colors = distinguishable_colors(length(allmethods) + 1);
methods_fixed_colors ={allmethods', methods_colors};

readpath = './result/ROC/';
% create a saving directory for the bar chart
savepath = './result/ROC_bar/'; 
if ~exist(savepath,'dir')
    mkdir(savepath);
end

%% load PRCurve.mat and draw PR curves
figure1 = figure('PaperPositionMode', 'auto' );

% Create axes
axes1 = axes('Parent', figure1, 'YTickLabel', {'0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'},...
    'YTick',[0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1],...
    'YGrid','on',...
    'XTickLabel',{'0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'},...
    'XTick',[0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1],...
    'XGrid','on',...
    'FontName','Arial',...
    'FontSize',12 );
hold on

for m = 1 : length(methods)
    prFileName = strcat(readpath,dataset, '_', methods{m}, '_ROCcurve.mat');
    R = load(prFileName);
    for k = 1 : 12
        input=R.ROC(1,k);
        TPR = input{1,1}(1, :);
        FPR = input{1,1}(2, :);
        score(m,k) = trapz(flip(FPR),flip(TPR));
        %plot(FPR,TPR, 'Color', methods_colors(m,:),'linewidth',2);

    end
    best_score_ind(m,1) = find(score(m,:)==max(score(m,:)));
    best_input=R.ROC(1,best_score_ind(m,1));
    best_TPR = best_input{1,1}(1, :);
    best_FPR = best_input{1,1}(2, :);
    best_score(m,1) = trapz(flip(best_FPR),flip(best_TPR));
    
    color_index=find(strcmp(methods{m},methods_fixed_colors{1}) == 1);
    plot(best_FPR,best_TPR, 'Color', methods_colors(color_index,:),'linewidth',2);
end

rounded_score = roundsd(score, 3);
ROC_score = roundsd(best_score, 3)

ROC_bar= {(methods)', ROC_score};
save([savepath dataset '_ROC_bar.mat'],'ROC_bar')

axis([0 1 0.2 1]);
hold off
grid on;

legend(methods, 'Location', 'SouthEast');
xlabel('False positive rate','fontsize',12);
ylabel('True positive rate','fontsize',12);
% for matlab 2015
%set(gca,'GridLineStyle',':','GridColor',[0 0 1])
%for matalb 2014 
set(gca,'GridLineStyle',':')
set(gca,'LooseInset',get(gca,'TightInset'))
%set(gcf, 'PaperSize', [6.0 4.6]');
box on
set(gca,'LooseInset',get(gca,'TightInset'))
print(gcf, '-depsc', ['ROC_' dataset '.eps'])

disp('************************** Done ******************************');

