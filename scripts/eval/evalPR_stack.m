%% A demo code to compute precision-recall curve for evaluating salient object rankning algorithms
% please cite our paper "Revisiting Salient Object Detection:
% Simultaneous Detection, Ranking, and Subitizing of Multiple Salient Objects",
% CVPR 2018, if you use the code in your research
% This code was originally modified from code provided for: "Contextual Hypergraph Modeling for Salient Object
% Detection", ICCV 2013
%% initialization

clear all ; close all ; clc;

method = 'RSDNET'; % name of the salient object method you want to evaluate, you need to change this
dataset = 'PASCAL-S'; % name of dataset, you need to change this

resultpath = ['../../predictions/saliency_maps_pascals_rsdnet-R/' '*.png']; % path to saliency maps, you might need to modify this

truthpath = ['../../data/PASCAL-S/gt/' '*.png']; % path to ground-truth masks, yoiu need to change this
savepath = './result/PRcurve/'; % save path of the 256 combinations of precision-recall values

if ~exist(savepath,'dir')
    mkdir(savepath);
end

dir_im = dir(resultpath);
assert(~isempty(dir_im),'No saliency map found, please check the path!');


imNum = length(dir_im);
precision_slice = zeros(256, 12);
recall_slice = zeros(256, 12);

match_table=[ 21 43 64 85 106 128 149 170 191 213 234 255];  % participants


%% compute pr curve

for i = 1 : imNum
    imName = dir_im(i).name;
    
    result_im = imread([resultpath(1:end-5),imName(1:end-4),resultpath(end-3:end)]);
   
    truth_im = imread([truthpath(1:end-5),imName]);
    truth_im = truth_im(:,:,1);
    result_im = result_im(:,:,1);
     
    for k = 1 : length(match_table)
        mask = truth_im;
        mask(find(truth_im(:) < match_table(k) )) = 0;
        mask(mask ~= 0) = 1;
        parfor threshold = 0:1:255
            index1 = (result_im >= threshold);
            truePositive = length(find(index1 & mask));
            % tp + fn
            groundTruth = length(find(mask));
            % tp + fp
            detected = length(find(index1));
            if truePositive ~= 0
                precision_slice(threshold + 1, k) = precision_slice(threshold + 1, k) + truePositive/detected;
                recall_slice(threshold + 1, k) = recall_slice(threshold + 1, k) + truePositive/groundTruth;
            end
        end
    end
    display(num2str(i));
end

precision_slice = precision_slice./imNum;
recall_slice = recall_slice./imNum;

 for k = 1 : length(match_table)
 	PR{k} = [(precision_slice(:, k))'; (recall_slice(:, k))'];
 end

save([savepath dataset '_' method '_PRCurve.mat'],'PR')

disp('****************************** Done ********************************');

