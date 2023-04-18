clear
close all
clc

% my_gt='D:\datasets\rellis_3D\Rellis-3D\cropped\00004\pylon_camera_node_label_color\';
% train_data = 'D:\datasets\rellis_3D\Rellis-3D\cropped\00004\pylon_camera_node\';
train_data = 'D:\datasets\rellis_3D\Rellis-3D\heatmap_gray_15\';
base_img = 'D:\datasets\rellis_3D\Rellis-3D\cropped\00000\pylon_camera_node\'

output = 'D:\datasets\rellis_3D\Rellis-3D\combine_img\'

path1 = train_data
path2 = base_img

myDir = path1; %gets directory
myFiles = dir(fullfile(myDir));

baseDir = base_img;
baseFiles = dir(fullfile(baseDir));


for k = 1:length(myFiles)
    if k>2
        FileName = myFiles(k).name;
        name = append(train_data,FileName);
        gimg = imread(name);

        baseFileName = baseFiles(k).name;
        base_name = append(base_img,baseFileName);
        bimg = imread(base_name);

        combine = bimg;
        combine(:,:,4) = gimg;

        baseFileName(find(baseFileName=='.',1,'last'):end) = []
        output_path = append(output,baseFileName);
        output_path = append(output_path,'.tif');
        imwrite(combine,output_path);

    end
end


%%




my_gt='D:\datasets\rellis_3D\Rellis-3D\cropped\00000\pylon_camera_node_label_color\';
train_data = 'D:\datasets\rellis_3D\Rellis-3D\cropped\00000\pylon_camera_node\';


path1 = train_data
path2 = my_gt

myDir = path1; %gets directory
myFiles = dir(fullfile(myDir));


myDirL = path2; %gets directory
myFilesL = dir(fullfile(myDirL));

M(length(myFiles))=zeros; %gets all wav files in struct
flag = 0;
for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    baseFileName(find(baseFileName=='.',1,'last'):end) = [];
    for j = 1:length(myFilesL)
        baseFileNameL = myFilesL(j).name;
        baseFileNameL(find(baseFileNameL=='.',1,'last'):end) = [];
        if(strcmp(baseFileName,baseFileNameL)==1)
            flag=1;

        end
    end
    if flag == 0
        fullFileName = fullfile(myDir, myFiles(k).name);
        delete(fullFileName)
    end
flag = 0;
end