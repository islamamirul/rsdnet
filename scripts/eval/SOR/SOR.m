%% A demo code to compute Salient Object Ranking (SOR) metric for evaluating salient object ranking algorithms
% please cite our paper "Revisiting Salient Object Detection:
% Simultaneous Detection, Ranking, and Subitizing of Multiple Salient Objects",
% CVPR 2018, if you use the code in your research

clear all; clc; close all;

% Path to the PASCAL-S ground truth saliency mask
gt_folder   = '../../../data/PASCAL-S/gt/';

% Path to the PASCAL-S ground truth segmentation mask
seg_mask_folder   = '../../../data/Pascal-S/seg_masks/';

% PATH to the prdection folder
pred_folder = '../../../predictions/saliency_maps_pascals_rsdnet-R/';

% model name
method = 'rsdnet';

predictions = dir([pred_folder '*.png']);

% outputs's saving path
savepath = './result/';

if ~exist(savepath,'dir')
    mkdir(savepath);
end

for j = 1 : length(predictions)
    
    fprintf('Processing image:%d/%d\n', j, length(predictions));
    
    img_name = predictions(j).name;
    sal_mask  = double(imread([gt_folder img_name(1:end-4) '.png']));
    pred  = double(imread([pred_folder img_name]));
    seg_mask  = double(imread([seg_mask_folder img_name(1:end-4) '.png']));
    unique_mask = (sal_mask).*(1+seg_mask);
    thresh = unique(unique_mask);

    for i = 2 : length(unique(unique_mask))
        gt_temp = unique_mask;
        gt_temp(gt_temp ~= thresh(i))= 0;
        gt_temp(gt_temp == thresh(i))= 255;

        ind = find(gt_temp > 0);
        regions_pixel = length(ind);
        total = sum(pred(ind));
        score(i-1,1) = total/regions_pixel;
        if length(unique(sal_mask(ind)))==1
            score(i-1,2) = unique(sal_mask(ind));
        else
            error('Saliency mask has multiple indexes')
        end
        clear gt_temp
        
    end
    rank{j,1} = score;
    rank{j,2} = img_name;
    rank{j,3} = Spearman_corrolation(rank{j,1}(:,1),rank{j,1}(:,2));

    clear score
    
end

disp('****************************** SAVING ********************************');

save([savepath, method, '.mat'], 'rank')


spear = zeros(length(rank),1);

for k = 1 : length(rank)
 spear(k,1) = rank{k,3};
end

ind_2 = find(~isnan(spear));
spear_valid = roundsd(spear(ind_2),6);
spear_norm = mat2gray(spear_valid);

SOR_score = roundsd(mean(spear_norm), 3)
