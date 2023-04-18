ptCloud = pcread('D:\datasets\rellis_3D\Rellis-3D\00000\os1_cloud_node_color_ply\frame000000-1581624652_770.ply');
%figure()
pcshow(ptCloud);

%%
down_img = imageDatastore('D:\datasets\rellis_3D\Rellis-3D\00000\pylon_camera_node\frame000000-1581624652_750.jpg')
I = readimage(down_img,1);

dist_coef = [-0.134313,-0.025905,0.002181,0.00084,0].'
img_width = 1920;
img_height = 1200;

%%
%camera info
camera_info = [2813.643275 2808.326079 969.285772 624.049972];
P = zeros(3,3);
P(1,1) = camera_info(1);
P(2,2) = camera_info(2);
P(3,3) = 1;
P(1,3) = camera_info(3);
P(2,3) = camera_info(4);
camera_mtx = P

%%
%LiDAR to cam mtx
%yaml = loadfile('D:\datasets\rellis_3D\Rellis_3D\00000\transforms.yaml');
w1 = -0.50507811;
x1 = 0.51206185;
y1 = 0.49024953;
z1 = -0.49228464;
x2 = -0.13165462;
y2 = 0.03870398;
z2 = -0.17253834;
w= -0.50507811;
x= 0.51206185;
y= 0.49024953;
z= -0.49228464;
q = [w x y z];
t = [x2 y2 z2];

R_vc = quat2rotm(q);

RT = eye(4,4);
RT = [R_vc,t.';0,0,0,1];
RT = inv(RT);


%%
%reshape camera
R_vc = RT(1:3,1:3);
T_vc = RT(1:3,4);
rvec = rotationMatrixToVector(R_vc).';
rvec = rvec * -1;
tvec = T_vc;

%%
%points filter
ctl = RT;
fov_x = 2*atan2(img_width, 2*P(1,1))*180/3.1415926+10;
fov_y = 2*atan2(img_height, 2*P(2,2))*180/3.1415926+10;
R = eye(4,4);
p_l = ones(131072,1);
p_l = [ptCloud.Location(:,1:3),p_l];
p_c = ctl*p_l.';
p_c = p_c.';
x = p_c(:,1);
y = p_c(:,2);
z = p_c(:,3);
dist = sqrt(x.^2 + y.^2 + z.^2).';
xangle = atan2(x,z)*180/pi;
yangle = atan2(y,z)*180/pi;
flag2 = (xangle > -fov_x/2) & (xangle < fov_x/2);
flag3 = (yangle > -fov_y/2) & (yangle < fov_x/2);
res = p_l(flag2&flag3,1:3);
x = res(:,1);
y = res(:,2);
z = res(:,3);
dist = sqrt(x.^2 + y.^2 + z.^2).';

color = ((dist-0)/(120-0)) * 120;

%%
%project points

%imagePoints = projectPoints(res, rvec, tvec, P)
%projectLidarPointsOnImage(ptCloud,[1200,1920],P);
%cv2.projectPoints(xyz_v[:,:],rvec, tvec, P, distCoeff)

%%
numr = 1200;
numc = 1920;
J = pointcloud2image( x,y,z,numr,numc );
C = imfuse(J,I)
figure()
imshow(C)
figure()
imshow(J)
%%
intrinsics = cameraIntrinsics([2813.643275 2808.326079],[969.285772 624.049972],[1200,1920])
tform = rigid3d(RT.')
[imPts,indices] = projectLidarPointsOnImage(ptCloud,intrinsics,tform);

figure
imshow(I)
hold on
plot(imPts(:,1),imPts(:,2),'.','Color','r')
hold off

%%
inten = zeros(length(indices),1);
for i = 1:length(indices)

if round(imPts(i,1)) <= 0
    imPts(i,1)= 1
end
if round(imPts(i,2)) <= 0
    imPts(i,2) = 1
end

if round(imPts(i,1)) >= 1920
    imPts(i,1)= 1920
end
if round(imPts(i,2)) >= 1200
    imPts(i,2) = 1200
end
inten(i) = (ptCloud.Intensity(indices(i)));
end
imPtsI = [imPts,inten];

%point_img = point_img.'

[n m k]=size(I);
%%
tic
[fulldepth depth] =dense_depth_map(imPtsI,n, m,18);
toc


figure;
seg_img = imagesc(fulldepth,[0 255]);
axis image
axis off
title('Full Depth map grid estimation');
%%
%write depth heatmap

cmapI = jet(700);
B = mat2gray(fulldepth);
B = gray2ind(B,700);
B = ind2rgb(B,cmapI);

filename = sprintf('%s_%d%s','D:\datasets\rellis_3D\Rellis-3D\heatmap_gray_15\',k,'.jpg');
%imwrite(B,filename)

%%

demo_img = imageDatastore('D:\datasets\rellis_3D\Rellis-3D\cropped\00000\pylon_camera_node\*.jpg');
ptCloud_pts = fullfile('D:\datasets\rellis_3D\Rellis-3D\cropped\00000\os1_cloud_node_color_ply', '*.ply');
theFiles = dir(ptCloud_pts);
%demo_img = readall(demo_img);
k=6234;
while k < 6235
    I = readimage(demo_img,k);
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName)
    ptcloud = pcread(fullFileName);

    [imPts,indices] = projectLidarPointsOnImage(ptcloud,intrinsics,tform);
    
     inten = zeros(length(indices),1);
    for i = 1:length(indices)
    
    if round(imPts(i,1)) <= 0
        imPts(i,1)= 1;
    end
    if round(imPts(i,2)) <= 0
        imPts(i,2) = 1;
    end
    
    if round(imPts(i,1)) >= 1920
        imPts(i,1)= 1920;
    end
    if round(imPts(i,2)) >= 1200
        imPts(i,2) = 1200;
    end
    inten(i) = (ptcloud.Intensity(indices(i)));
    end
    imPtsI = [imPts,inten];
    
    [n m z]=size(I);
    
    

     [fulldepth depth] =dense_depth_map(imPtsI,n, m,4);

     cmapI = jet(700);
     B = mat2gray(fulldepth);
     B = gray2ind(B,700);
     B = ind2rgb(B,cmapI);

%     imshow(B)
    
     
     filename = sprintf('%s%s%s','D:\datasets\rellis_3D\Rellis-3D\heatmap_color_15\',baseFileName(1:end-4),'.jpg');
     %imwrite(B,filename)
    k = k + 1;
end