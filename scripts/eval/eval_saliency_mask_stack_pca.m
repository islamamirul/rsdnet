%%   Principle Component Analysis for Stacked mask
%  please cite our paper "Revisiting Salient Object Detection:
% Simultaneous Detection, Ranking, and Subitizing of Multiple Salient Objects",
% CVPR 2018, if you use the code in your research


clear ; clc; close all;


gt_folder = './data/PASCAL-S/stacked_seg_masks/'; % generated ground truth stack
pred_folder = './predictions/pascal-s/predicted_saliency_stack/'; % predicted stack by NRSS

imagepath = ['./data/Pascal-S/images/' '*.png']; % image to test
truthpath = ['./data/Pascal-S/gt_original/' '*.png']; % grounf truth for the image

pred_images = dir([pred_folder '*.mat']);

size_subject = 12;

match_table=[ 21 43 64 85 106 128 149 170 191 213 234 255];

nameofimages = {'449.mat', '114.mat'};

% PlasmaColorMap = (plasma(256));
% colormap(PlasmaColorMap)
PlasmaColorMap = parula(256);

savepath = './result/stack_analysis/';


for k = 1 : length(nameofimages)
    img_name = nameofimages{k};

    truth_im = imread([truthpath(1:end-5),img_name(1:end-4) '.png']);
    image = imread([imagepath(1:end-5),img_name(1:end-4),'.jpg']);
    
    % load and save predicted stack (if required)
    figure(1)
    pred=load([pred_folder img_name]);
    pred = pred.slice;
    pred= permute(pred,[2  3 1]);
    for i=1:size_subject
        min_max_slice(i,1) = min(min(pred(:,:,i)));
        min_max_slice(i,2) = max(max(pred(:,:,i)));
        subplot(4,3,i);
        imshow(pred(:,:,i),[])
        %colormap(PlasmaColorMap)
        
        %imwrite(mat2gray(pred(:,:,i)),PlasmaColorMap,[savepath imName(1:end-4) '_slice_' num2str(i) '.png'])
    end
        title (img_name)
    
    % load and save Gt stack
    figure(2)
    gt=load([gt_folder img_name]);
    gt = gt.stacked_masks;
    for i=1:12
        subplot(4,3,i);
        imshow(gt(:,:,i),[])
        colormap(PlasmaColorMap)
        imwrite(uint8(gt(:,:,i)),PlasmaColorMap,[savepath img_name(1:end-4) '_gt_slice_' num2str(i) '.png'])

    end
    
    % save image and its Gt
    imwrite(truth_im,PlasmaColorMap,[savepath img_name(1:end-4) '_GT' '.png'])
    imwrite(image,[savepath img_name(1:end-4) '_image' '.png'])
    
    % calculate and save PCA
    linear_pred = reshape(pred,[size(pred,1)*size(pred,2) size(pred,3)]);

    [coeff,score,latent] = pca(linear_pred);
    
    post_pred = reshape(score,[size(pred,1) size(pred,2) size(pred,3)]);
    new_pred= zeros([size(pred,1) size(pred,2) 3]);
    new_pred(:,:,1)=mat2gray(post_pred(:,:,1));
    new_pred(:,:,2)=mat2gray(post_pred(:,:,2));
    new_pred(:,:,3)=mat2gray(post_pred(:,:,3));

   figure,  imshow(new_pred,[])
   
    title (img_name)
        fprintf('Processing image:%d/%d\n',k, length(nameofimages));
        
   imwrite(uint8(255*mat2gray(new_pred)),[savepath img_name(1:end-4) '_pca' '.png'])
end





