function p1c()
% EE569 Homework #2
% Name: Agam Sidhu
% USC ID: 3027948957
% USC Email: agamsidh@usc.edu
% Submission Date: February 22, 2026
% Problem 1(c): Structured Edge Detection with SE-Forest
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
percent = [0.15 0.20 0.25];

for img= images
    imageRun(img{1}, inp, outp, model, percent);
end
end

function imageRun(filename, inp, outp, model, percent)
rgb = imread(fullfile(inp, filename));
iVal = single(rgb)/255;
edgeProb = edgesDetect(iVal, model);
[~,base] = fileparts(filename);
imwrite(uint8(edgeProb * 255), fullfile(outp, sprintf('%s_SE_prob.png', base)));
for t = percent
    binary = edgeProb >= t;
    imwrite(uint8(binary) * 255, fullfile(outp, sprintf('%s_SE_bin_T%.2f.png', base, t)));
end
fprintf('Saved %s\n', base);
end