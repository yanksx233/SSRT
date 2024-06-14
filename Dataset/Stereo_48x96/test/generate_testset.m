clc
clear
%% KITTI2012
img_list = dir('./ETH3D_hr/hr');
for idx_file = 3:length(img_list)
    file_name = img_list(idx_file).name;
    % idx_name = find(file_name == '_');
    % file_name = file_name(1:idx_name-1);
    
    img_0 = imread(['./ETH3D_hr/hr/',img_list(idx_file).name, '/hr0.png']);
    img_1 = imread(['./ETH3D_hr/hr/',img_list(idx_file).name, '/hr1.png']);
    img_0 = imresize(img_0, 1/2, 'bicubic');
    img_1 = imresize(img_1, 1/2, 'bicubic');
    
    %% x4
    scale_list = [4, 2];
    for idx_scale = 1:length(scale_list)
        scale = scale_list(idx_scale);
        
        %% generate HR & LR images
        img_hr_0 = modcrop(img_0);
        img_hr_1 = modcrop(img_1);
        img_lr_0 = imresize(img_hr_0, 1/scale, 'bicubic');
        img_lr_1 = imresize(img_hr_1, 1/scale, 'bicubic');
        
        mkdir('./ETH3D/hr');
        mkdir(['./ETH3D/hr/', file_name]);
        mkdir(['./ETH3D/lr_x', num2str(scale)]);
        mkdir(['./ETH3D/lr_x', num2str(scale), '/', file_name]);
        
        imwrite(img_hr_0, ['./ETH3D/hr/', file_name, '/hr0.png']);
        imwrite(img_hr_1, ['./ETH3D/hr/', file_name, '/hr1.png']);
        imwrite(img_lr_0, ['./ETH3D/lr_x', num2str(scale), '/', file_name, '/lr0.png']);
        imwrite(img_lr_1, ['./ETH3D/lr_x', num2str(scale), '/', file_name, '/lr1.png']);
    end
end