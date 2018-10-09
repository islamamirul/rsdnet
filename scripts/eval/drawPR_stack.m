%% A demo code to save precision-recall curve and pr bars for evaluating salient object ranking algorithms
% please cite our paper "Revisiting Salient Object Detection:
% Simultaneous Detection, Ranking, and Subitizing of Multiple Salient Objects",
% CVPR 2018, if you use the code in your research
% This code was originally modified from code provided for: "Contextual Hypergraph Modeling for Salient Object
% Detection", ICCV 2013

clear all; close all;
dataset = 'PASCAL-S'; % name of the dataset

%% to draw
allmethods = {'DSR','HS','MC','MDF','MTDS','DHS','NLDF','DSS','AMULET','UCF','RSDNET-R'};  % you can add more methods separated by comma
methods =  {'DSR','HS','MC','MDF','MTDS','DHS','NLDF','DSS','AMULET','UCF','RSDNET-R'};

allmethods = {'RSDNET'}; % you can add more names of methods separated by comma
methods =  {'RSDNET'};

methods_colors = distinguishable_colors(length(allmethods) + 1);
methods_fixed_colors = {allmethods',methods_colors};

readpath = './result/PRcurve/';
savepath = './result/FM_bar/';

if ~exist(savepath,'dir')
    mkdir(savepath);
end

%% load PRCurve.mat and draw PR curves
figure1 = figure('PaperPositionMode', 'auto' );

% Create axes
axes1 = axes('Parent',figure1,'YTickLabel',{'0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'},...
    'YTick',[0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1],...
    'YGrid','on',...
    'XTickLabel',{'0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'},...
    'XTick',[0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1],...
    'XGrid','on',...
    'FontName','Arial',...
    'FontSize',12 );
hold on

for m = 1 : length(methods)
    prFileName = strcat(readpath,dataset, '_', methods{m}, '_PRCurve.mat');
    R = load(prFileName);
    for k= 1 :12
        input=R.PR(1,k);
        precision = input{1,1}(1, :);
        recall = input{1,1}(2, :);
        
        %% those condition are just for better visulization of the curve
        if max(strcmp(methods{m},'DHS')) == 1
            precision = precision(1:end-1);
            recall =recall(1:end-1)  ;
        end
        
        if max(strcmp(methods{m},'DRFI')) == 1
            precision = precision(1:end-8);
            recall =recall(1:end-8)  ;
        end
        
        if max(strcmp(methods{m},'ELD')) == 1
            precision = precision(1:end-1);
            recall =recall(1:end-1)  ;
        end
        
        if max(strcmp(methods{m},'rsdnet')) == 1
            precision = precision(1:end-2);
            recall =recall(1:end-2)  ;
        end
        %%
        for j = 1:length(precision)
            fmeasure_slice(j,k) = 1.3*precision(j)*recall(j)/(0.3*precision(j)+recall(j));
        end
    end
    max_fmeasure_slice_save{m,1} = roundsd(max(fmeasure_slice),3);
    
    max_fmeasure_slice= max(fmeasure_slice);
    max_fmeasure(m,1) = max(max_fmeasure_slice);
    average_fmeasure_slice= mean(fmeasure_slice);
    average_fmeasure(m,1)= max(average_fmeasure_slice);
    median_fmeasure_slice= median(fmeasure_slice);
    median_fmeasure(m,1)= max(median_fmeasure_slice);
    
    best_fscore_ind(m) = find(max_fmeasure_slice== max(max_fmeasure_slice));
    best_input=R.PR(1,best_fscore_ind(m));
    best_precision = best_input{1,1}(1, :);
    best_recall = best_input{1,1}(2, :);
    
    if max(strcmp(methods{m},'DHS')) == 1
        best_precision = best_precision(1:end-1);
        best_recall =best_recall(1:end-1)  ;
    end
    
    if max(strcmp(methods{m},'DRFI')) == 1
        best_precision = best_precision(1:end-1);
        best_recall =best_recall(1:end-1)  ;
    end
    
    if max(strcmp(methods{m},'ELD')) == 1
        best_precision = best_precision(1:end-6);
        best_recall =best_recall(1:end-6)  ;
    end
    
    if max(strcmp(methods{m},'rsdnet')) == 1
        best_precision = best_precision(1:end-2);
        best_recall =best_recall(1:end-2)  ;
    end
    
    color_index=find(strcmp(methods{m},methods_fixed_colors{1}) == 1);
    plot(best_recall,best_precision, 'Color', methods_colors(color_index,:),'linewidth',2);
    
end

% might be needed
% for r = 1 : length(methods)
%     for c = 1 : 12
%         max_fmeasure_slice_save2(r, c) = max_fmeasure_slice_save{r, 1}(c);
%     end
% end

max_fmeasure_score = roundsd(max_fmeasure, 3)
average_fmeasure_score = roundsd(average_fmeasure, 3)
median_fmeasure_score = roundsd(median_fmeasure, 3)

FM_bar= {(methods)', max_fmeasure_score};

save([savepath dataset '_FM_bar.mat'],'FM_bar')

axis([0 1 0.3 1]);
hold off
grid on;
legend(methods, 'Location', 'SouthWest');
xlabel('Recall','fontsize',12);
ylabel('Precision','fontsize',12);
% for matlab 2015
%set(gca,'GridLineStyle',':','GridColor',[0 0 1])
%for matalb 2014 
set(gca,'GridLineStyle',':')
set(gca,'LooseInset',get(gca,'TightInset'))
box on
set(gca,'LooseInset',get(gca,'TightInset'))
print(gcf, '-depsc', ['PR_' dataset '.eps'])

disp('************************** Done ******************************');



