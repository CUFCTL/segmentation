
resnet18();


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

condensed_classes = [
"sky"
"traversable"
"non-traversable"
"obstacles"
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

condensed_classes = [
"sky"
"traversable"
"nonTraversable"
"obstacles"
];

condensed_labelIDs = { ...
    % "sky
    [
    0 0 255; ... % "sky
    ] 

    % "traversable"
    [
    0 102 0; ... % grass
    108 64 20; ... % "dirt"
    64 64 64; ... % "asphalt"
    170 170 170; ... % "concrete"
    134 255 239; ... % "puddle"
    99 66 34; ... % "mud"
    0 128 255; ... % "water"
    ]

    % "non-traversable"
    [
    
    0 0 0; ... % "void"
    110 22 138; ... % "rubble"
    255 153 204; ... % "bush"
    
    ]
    
    % "obstacles"
    [
    255 255 0; ... % "vehicle"
    41 121 255; ... % "barrier"
    102 0 0; ... % "log"
    0 153 153; ... %pole
    255 0 127; ... % "object"
    255 0 0; ... % "building"
    204 153 255; ... % "person"
    102 0 204; ... % "fence"
    0 255 0; ... % "Tree"

    ]
};

condensed_cmap = [
    255 0 0 % "sky"
    0 255 0 % "traversable"
    0 0 255 % "non-traversable"
    255 255 0 % "obstacles"
];

condensed_cmap = condensed_cmap ./ 255;


classes = condensed_classes;
labelIDs = condensed_labelIDs;
cmap = condensed_cmap;
%%
%my_gt = 'C:\Users\max\Documents\MATLAB\vipr\resize_labels\'
% my_gt='D:\datasets\rellis_3D\Rellis-3D\00000\pylon_camera_node_label_color\';
% train_data = 'C:\Users\max\Documents\MATLAB\vipr\resize_train/'
% train_data = 'D:\datasets\rellis_3D\Rellis-3D\00000\pylon_camera_node\';
%labelIDs = pixID();
my_gt='/media/eceftl11/a41491ac-71f0-4b6e-9240-033ae2892263/vipr/Rellis-3D/cropped/00000/pylon_camera_node_label_color/';
%train_data='/media/eceftl11/a41491ac-71f0-4b6e-9240-033ae2892263/vipr/Rellis-3D/cropped/00000/pylon_camera_node/';

train_data = '/media/eceftl11/a41491ac-71f0-4b6e-9240-033ae2892263/vipr/Rellis-3D/combine_img';

imds = imageDatastore(train_data);
countEachLabel(imds);


numClasses = numel(classes);
imageSize = [1200 1920 4];
labelDir = fullfile(my_gt);
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);



%cds = combine(imds,pxds);

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

% input_layer = imageInputLayer([1200 1920 4])
% lgraph = replaceLayer(lgraph,'data',input_layer)



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
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'ValidationData',pximdsVal,...
    'MaxEpochs',50, ...  
    'MiniBatchSize',3, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 10, ...
    'ExecutionEnvironment','gpu');

%%

%lgraph_1 = load('lgraph_1.mat');



%rcds = read(cds)
%%

doTraining = true;
if doTraining    
    [net, info] = trainNetwork(pximds,lgraph_1,options);
else
    data = load('rellis_first_85.mat'); 
    net = data.net;
end



%%

%I = readimage(imdsTest,25);
I = readimage(imdsTest,45);
C = semanticseg(I, net);
%se = offsetstrel('ball',5,5);
se = strel('rectangle',[40 30]);

I = I(:,:,1:3);

B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.5);
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
    'MiniBatchSize',3, ...
    'WriteLocation',tempdir, ...
    'Verbose',false);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);

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

% demo_img = imageDatastore('D:\datasets\rellis_3D\Rellis-3D\00000\pylon_camera_node\*.jpg');
% %demo_img = readall(demo_img);
% k=1;
% while k < 2848
% I = readimage(demo_img,k);
% C = semanticseg(I, net);
% B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.1);
% filename = sprintf('%s_%d%s','D:\datasets\rellis_3D\Rellis-3D\seg_demo\frame',k,'.jpg');
% imwrite(B,filename)
% k = k + 1;
% end
%%
%demo_img = imageDatastore('D:\datasets\rellis_3D\Rellis-3D\00000\pylon_camera_node\*.jpg');
demo_img = imageDatastore('/home/eceftl11/vipr/spring22/frames_demo/*.jpg')
%demo_img = readall(demo_img);
k=314;
while k < 314
I = readimage(demo_img,k);
I = imresize(I,[1200,1920]);
C = semanticseg(I, net);
B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.5);
filename = sprintf('%s_%d%s','/home/eceftl11/vipr/spring22/demo_seg/',k,'.jpg');
%imwrite(B,filename)
k = k + 1;
end
%%
down_img = imageDatastore('/home/eceftl11/frame000004-1581624075_649.jpg')
I = readimage(down_img,1);
I = imresize(I,[1200 1920]);
tic
C = semanticseg(I, net);
toc
%se = offsetstrel('ball',5,5);
se = strel('rectangle',[40 30]);

B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.01);
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

%figure
%imshow(I)
%%
metrics
metrics.ClassMetrics
metrics.ConfusionMatrix
metrics.NormalizedConfusionMatrix
order = ["sky";"traversable";"nontraversable";"obstacles"]
order = categorical(order)
conf = table2array(metrics.ConfusionMatrix)

figure
cm = confusionchart(conf,order)
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = 'RGB-Depth Confusion Matrix';
