% Exploring Rellis-3D Lidar Dataset

home_dir = '../../../..';
LIDAR_PATH = fullfile(home_dir,'datasets','Rellis-3D');

% Get folder contents and remove the linux '.' and '..' dirs
video_dir = dir(LIDAR_PATH);
video_dir = video_dir(~ismember({video_dir.name},{'.','..'}));
% Loop through lidar sub directories and get total number of lidar files
num_files = 0;
for i=1 : numel(video_dir)
    subdirs = dir(fullfile(LIDAR_PATH, video_dir(i).name, 'os1_cloud_node_color_ply'));
    subdirs = subdirs(~ismember({subdirs.name},{'.','..'}));
    num_files = num_files + numel(subdirs);
end

% Create array to store all file paths of lidar files
% then loop through the directories again to get these paths
lidar_files = strings([1, num_files]);
file_index = 1;
for i=1 : numel(video_dir)
    subdirs = fullfile(LIDAR_PATH, video_dir(i).name, 'os1_cloud_node_color_ply');
    subdirs_files = dir(subdirs);
    subdirs_files = subdirs_files(~ismember({subdirs_files.name},{'.','..'}));
    for j=1 : numel(subdirs_files)
        lidar_files(1, file_index) = fullfile(subdirs, subdirs_files(j).name);
        file_index = file_index + 1;
    end
end

figure();
%subplot(2,2,[2,4])
file_num = 1345;
point_cloud_a = pcread(lidar_files(file_num));
pt_cloud_ax = pcshow(point_cloud_a)
view(pt_cloud_ax, 80, 10) % set camera angle
zoom(pt_cloud_ax, 5) % zoom into 3d plot by factor of 5
title(lidar_files(file_num))
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')