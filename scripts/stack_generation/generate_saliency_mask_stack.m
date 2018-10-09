% Generate the ground truth stack of saliency masks from the original
% PASCAL-S ground truth saliency map

clear all;clc;close all;

% root folder to the PASCAL-S saliency GT

gt_folder = '../../data/PASCAL-S/gt/';
gt_images = dir([gt_folder '*.png']);

% folder where to save the results

save_folder = '../../data/PASCAL-S/stacked_seg_masks/';

if ~exist(save_folder,'dir')
    mkdir(save_folder);
end


size_subject = 12;  % total observers participated in the labelling process for PASCAL-S

match_table = [ 21 43 64 85 106 128 149 170 191 213 234 255];

for i = 1 : length(gt_images)
    
    fprintf('Processing image:%d/%d\n', i, length(gt_images));
    
    img_name = gt_images(i).name;
    gt  = imread([gt_folder img_name]);
    unique_objects = unique(gt);
    stacked_masks = zeros (size(gt,1), size(gt,2), size_subject);
    
    for k = 1 : length(match_table)
        mask = gt;
        mask(find(gt(:) < match_table(k) )) = 0;
        stacked_masks (:, :, k) =  mask;
    end
    
    stacked_masks(stacked_masks > 0) = 1;
    save ([save_folder img_name(1 : end-4) '.mat'], 'stacked_masks' );
end
