%% A demo code to compute mean Mean Absolute Error (MAE) for evaluating salient object ranking algorithms
% please cite our paper "Revisiting Salient Object Detection:
% Simultaneous Detection, Ranking, and Subitizing of Multiple Salient Objects",
% CVPR 2018, if you use the code in your research
% This code was originally modified from code provided for: "Contextual Hypergraph Modeling for Salient Object
% Detection", ICCV 2013
% MAE is proposed by "Saliency Filters: Contrast Based Filtering for
% Salient Region Detection", CVPR 2012


clear all ; close all ; clc;

method = 'RSDNET'; % name of the salient object method you want to evaluate, you need to change this
dataset = 'PASCAL-S'; % name of dataset, you need to change this

resultpath = ['../../predictions/saliency_maps_pascals_rsdnet/' '*.png']; % path to saliency maps, you might need to modify this

truthpath = ['../../data/PASCAL-S/gt/' '*.png']; % path to ground-truth masks, yoiu need to change this
savepath = './result/MAE/'; % save path of the 256 combinations of precision-recall values

if ~exist(savepath,'dir')
    mkdir(savepath);
end

dir_im = dir(resultpath);
assert(~isempty(dir_im),'No saliency map found, please check the path!');

imNum = length(dir_im);
MAE = zeros(1, 12);

match_table=[ 21 43 64 85 106 128 149 170 191 213 234 255];

%% compute MAE

for i = 1 : imNum
    imName = dir_im(i).name;
    
    result_im = imread([resultpath(1 : end-5),imName(1 : end-4),resultpath(end-3 : end)]);
    result_im = im2double(result_im(:, :, 1));
    truth_im = imread([truthpath(1 : end-5),imName]);
    truth_im = double(truth_im(:, :, 1));
    
    
    for k=1:length(match_table)
        mask = truth_im;
        mask(find(truth_im(:) < match_table(k) )) = 0;
        mask(mask ~= 0) = 1;
        MAE(1, k) = MAE(1 ,k) + mean2(abs(mask - result_im));
    end
    
    display(num2str(i));
end

MAE = MAE./imNum;

best_MAE = min(MAE);
MAE_score = roundsd(best_MAE, 2)
best_MAE_ind = find(MAE == min(MAE));
          
save([savepath dataset '_' method '_MAE'], 'MAE');


