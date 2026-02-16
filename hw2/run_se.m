function run_se()
clc; clear; close all;

edgePath = fileparts(which('edgesDetect'));
modelPath = fullfile(edgePath,'models','forest','modelBsds');

m = load(modelPath);
model = m.model;
model.opts.nms = 1;

inp  = 'input';
outp = 'output';
mkdir(outp);
images = {'Bird.jpg','Deer.jpg'};
percent = 0.20;

for img= images
    imageRun(img{1}, inp, outp, model, percent);
end
end

function imageRun(filename, inp, outp, model, percent)
rgb = imread(fullfile(inp, filename));
iVal = single(rgb)/255;
edgeProb = edgesDetect(iVal, model);
binary = edgeProb >= percent;
[~,base] = fileparts(filename);

imwrite(uint8(edgeProb*255), fullfile(outp, sprintf('%s_SE_prob.png', base)));
imwrite(uint8(binary)*255,   fullfile(outp, sprintf('%s_SE_bin_T%.2f.png', base, percent)));
fprintf('Saved %s\n', base);
end