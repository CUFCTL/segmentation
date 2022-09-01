
resnet18();

pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/deeplabv3plusResnet18CamVid.mat';
pretrainedFolder = fullfile(tempdir,'pretrainedNetwork');
pretrainedNetwork = fullfile(pretrainedFolder,'deeplabv3plusResnet18CamVid.mat'); 
if ~exist(pretrainedNetwork,'file')
    mkdir(pretrainedFolder);
    disp('Downloading pretrained network (58 MB)...');
    websave(pretrainedNetwork,pretrainedURL);
end

imageURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip';
labelURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip';
 
outputFolder = fullfile(tempdir,'CamVid'); 
labelsZip = fullfile(outputFolder,'labels.zip');
imagesZip = fullfile(outputFolder,'images.zip');

if ~exist(labelsZip, 'file') || ~exist(imagesZip,'file')   
    mkdir(outputFolder)
       
    disp('Downloading 16 MB CamVid dataset labels...'); 
    websave(labelsZip, labelURL);
    unzip(labelsZip, fullfile(outputFolder,'labels'));
    
    disp('Downloading 557 MB CamVid dataset images...');  
    websave(imagesZip, imageURL);       
    unzip(imagesZip, fullfile(outputFolder,'images'));    
end


imgDir = fullfile(outputFolder,'images','701_StillsRaw_full');
imds = imageDatastore(imgDir);
I = readimage(imds,559);
I = histeq(I);
imshow(I)



classes = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
    ];


labelIDs = camvidPixelLabelIDs();
labelDir = fullfile(outputFolder,'labels');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);

C = readimage(pxds,559);
cmap = camvidColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);
figure
imshow(B)
pixelLabelColorbar(cmap,classes);
tbl = countEachLabel(pxds)
frequency = tbl.PixelCount/sum(tbl.PixelCount);

%%
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds,pxds);
numTrainingImages = numel(imdsTrain.Files)

numValImages = numel(imdsVal.Files)
numTestingImages = numel(imdsTest.Files)
% Specify the network image size. This is typically the same as the traing image sizes.

imageSize = [720 960 3];
% Specify the number of classes.
numClasses = numel(classes);

lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet18");

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);

pximdsVal = pixelLabelImageDatastore(imdsVal,pxdsVal);


% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.7, ...
    'InitialLearnRate',1e-2, ...
    'L2Regularization',0.005, ...
    'ValidationData',pximdsVal,...
    'MaxEpochs',10, ...  
    'MiniBatchSize',8, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 4, ...
    'ExecutionEnvironment','gpu');


augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain, ...
    'DataAugmentation',augmenter);


doTraining = false;
if doTraining    
    [net, info] = trainNetwork(pximds,lgraph,options);
else
    data = load(pretrainedNetwork); 
    netRes = data.net;
end

%%
I = readimage(imdsTest,15);
C = semanticseg(I, net);

B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
figure
imshow(B)
pixelLabelColorbar(cmap, classes);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Rellis 3D train


classes = [
"void"
"dirt"
"grass"
"tree"
"pole"
"water"
"sky"
"vehicle"
"object"
"asphalt"
"building"
"log"
"person"
"fence"
"bush"
"concrete"
"barrier"
"puddle"
"mud"
"rubble"
];

    
labelIDs = { ...

    [
    0 0 0; ... % "void"
    ]

    [
    108 64 20; ... % "dirt"
    ]

    [
    0 102 0; ... % "grass"
    ]
    
    [
    0 255 0; ... % "Tree"
    ]
    
    [
    0 153 153; ... %pole
    ]

    [
    0 128 255; ... % "water"
    ]

    [
    0 0 255; ... % "sky"
    ]

    [
    255 255 0; ... % "vehicle"
    ]

    [
    255 0 127; ... % "object"
    ]

    [
    64 64 64; ... % "asphalt"
    ]

    [
    255 0 0; ... % "building"
    ]

    [
    102 0 0; ... % "log"
    ]

    [
    204 153 255; ... % "person"
    ]

    [
    102 0 204; ... % "fence"
    ]

    [
    255 153 204; ... % "bush"
    ]

    [
    170 170 170; ... % "concrete"
    ]

    [
    41 121 255; ... % "barrier"
    ]

    [
    134 255 239; ... % "puddle"
    ]

    [
    99 66 34; ... % "mud"
    ]

    [
    110 22 138; ... % "rubble"
    ]
    
};

cmap = [
    0 0 0 % "void"
    108 64 20 % "dirt"
    0 102 0 % "grass"
    0 255 0 % "Tree"
    0 153 153 %pole
    0 128 255 % "water"
    0 0 255 % "sky"
    255 255 0 % "vehicle"
    255 0 127 % "object"
    64 64 64 % "asphalt"
    255 0 0 % "building"
    102 0 0 % "log"
    204 153 255 % "person"
    102 0 204 % "fence"
    255 153 204 % "bush"
    170 170 170 % "concrete"
    41 121 255 % "barrier"
    134 255 239 % "puddle"
    99 66 34 % "mud"
    110 22 138 % "rubble"
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;


%my_gt = 'C:\Users\max\Documents\MATLAB\vipr\resize_labels\'
% my_gt='D:\datasets\rellis_3D\Rellis-3D\00000\pylon_camera_node_label_color\';
% train_data = 'C:\Users\max\Documents\MATLAB\vipr\resize_train/'
% train_data = 'D:\datasets\rellis_3D\Rellis-3D\00000\pylon_camera_node\';
%labelIDs = pixID();
my_gt='D:\datasets\rellis_3D\Rellis-3D\cropped\00000\pylon_camera_node_label_color\';
train_data = 'D:\datasets\rellis_3D\Rellis-3D\cropped\00000\pylon_camera_node\';

imds = imageDatastore(train_data);
countEachLabel(imds);
I = readimage(imds,3);
C = semanticseg(I, net);
B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
figure
imshow(B)
pixelLabelColorbar(cmap, classes);
I = histeq(I);
figure
imshow(I)


numClasses = numel(classes);
imageSize = [1200 1920 3];
labelDir = fullfile(my_gt);
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);



cds = combine(imds,pxds);


C = readimage(pxds,3);
%cmap = camvidColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);
figure
imshow(B)

tbl = countEachLabel(pxds);


if(tbl.ImagePixelCount(3)==0)
    tbl.ImagePixelCount(3)=1;
end
if(tbl.PixelCount(3)==0)
    tbl.PixelCount(3)=1;
end


[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] =parseID(imds,pxds);
numTrainingImages = numel(imdsTrain.Files);
numValImages = numel(imdsVal.Files);
numTestingImages = numel(imdsTest.Files);


% Specify the network image size. This is typically the same as the traing image sizes.
%%

% Specify the number of classes.
numClasses = numel(classes);

% Create DeepLab v3+.
lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet18");

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);

% Define validation data.
pximdsVal = pixelLabelImageDatastore(imdsVal,pxdsVal);

augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain, ...
    'DataAugmentation',augmenter);


options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.7, ...
    'InitialLearnRate',1e-2, ...
    'L2Regularization',0.005, ...
    'ValidationData',pximdsVal,...
    'MaxEpochs',50, ...  
    'MiniBatchSize',3, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 6, ...
    'ExecutionEnvironment','gpu');



%rcds = read(cds)
%%

doTraining = true;
if doTraining    
    [net, info] = trainNetwork(pximds,lgraph,options);
else
    data = load('rellis_first_85.mat'); 
    net = data.net;
end



%%

I = readimage(imdsTest,7);
C = semanticseg(I, net);
%se = offsetstrel('ball',5,5);
se = strel('rectangle',[40 30]);

B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.1);
%B = imerode(B,se);
%B = imerode(B,se);
%B = imdilate(B,se);
%B = imdilate(B,se);
%B_g = im2gray(B)
%B_g = wiener2(B_g,[5,5]);
%B = cat(3, B_g, B_g, B_g)
figure
imshow(B)
pixelLabelColorbar(cmap, classes);

%%
expectedResult = readimage(pxdsTest,35);
actual = uint8(C);
expected = uint8(expectedResult);
figure
imshowpair(actual, expected)

%%
iou = jaccard(C,expectedResult);
table(classes,iou)
%%
pxdsResults = semanticseg(imdsTest,net, ...
    'MiniBatchSize',4, ...
    'WriteLocation',tempdir, ...
    'Verbose',false);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
%%
metrics.DataSetMetrics

metrics.ClassMetrics

%%
%timing
tic
k=1;
while k < 1248
I = readimage(imdsTest,k);
C = semanticseg(I, net);
k = k + 1;
end
full_time = toc
average_time = full_time/1247
%%

demo_img = imageDatastore('D:\datasets\rellis_3D\Rellis-3D\00000\pylon_camera_node\*.jpg');
%demo_img = readall(demo_img);
k=1;
while k < 2848
I = readimage(demo_img,k);
C = semanticseg(I, net);
B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.1);
filename = sprintf('%s_%d%s','D:\datasets\rellis_3D\Rellis-3D\seg_demo\frame',k,'.jpg');
imwrite(B,filename)
k = k + 1;
end
%%
demo_img = imageDatastore('D:\datasets\rellis_3D\Rellis-3D\00000\pylon_camera_node\*.jpg');
%demo_img = readall(demo_img);
k=1;
while k < 2848
I = readimage(demo_img,k);
%C = semanticseg(I, net);
%B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.1);
filename = sprintf('%s_%d%s','D:\datasets\rellis_3D\Rellis-3D\00000\renamed\frame',k,'.jpg');
imwrite(I,filename)
k = k + 1;
end