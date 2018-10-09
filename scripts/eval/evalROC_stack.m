%% A demo code to compute ROC curve for evaluating salient object ranking algorithms
%% please cite our paper "Revisiting Salient Object Detection:
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
savepath = './result/ROC/';

if ~exist(savepath,'dir')
    mkdir(savepath);
end

dir_im = dir(resultpath);
assert(~isempty(dir_im),'No saliency map found, please check the path!');

imNum = length(dir_im);
TPR_slice = zeros(256,12);
FPR_slice = zeros(256,12);

match_table=[ 21 43 64 85 106 128 149 170 191 213 234 255];  % participants


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
        %figure(), imshow(mask,[])
        P = sum(sum(mask));
        N = sum(sum(~mask));
        parfor threshold = 0:1:255
            if P ~= 0 && N ~= 0
                index1 = (result_im>=threshold);
                TP = sum(sum(mask & index1));
                FP = sum(sum((~mask) & index1));
                % commulative summation to get the TPR & FPR for the whole dataset
                TPR_slice(threshold + 1, k) = TPR_slice(threshold + 1, k) + TP/P;
                FPR_slice(threshold + 1, k) = FPR_slice(threshold + 1, k) + FP/N;
            end
        end
    end
    display(num2str(i));
end

%get the mean TPR& FPR

TPR_slice = TPR_slice./imNum;
FPR_slice = FPR_slice./imNum;

index = (TPR_slice > 1);
TPR_slice(index) = 1;
index = (FPR_slice > 1);
FPR_slice(index) = 1;

for k = 1 : length(match_table)
    ROC{k} = [(TPR_slice(:, k))'; (FPR_slice(:, k))'];
end

save([savepath dataset '_' method '_ROCcurve.mat' ],'ROC')

disp('************************** Done ******************************');

