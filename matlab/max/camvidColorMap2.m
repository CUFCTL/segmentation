function cmap2 = camvidColorMap2()
% Define the colormap used by CamVid dataset.

cmap2 = [
    128 128 128   % Sky
    128 64 128    % Road
    128 128 0     % Tree
    192 128 128   % Grass
    64 64 128     % Vegatation
    64 0 128      % Obstacle
    ];

% Normalize between [0 1].
cmap2 = cmap2 ./ 255;
end