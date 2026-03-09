function p1d(imgSel)
% EE569 Homework #2
% Name: Agam Sidhu
% USC ID: 3027948957
% USC Email: agamsidh@usc.edu
% Submission Date: February 22, 2026
% Problem 1(d): Evaluation of Edge Detectors against Human GT
if nargin < 1, imgSel = "ALL"; end
imgSel = string(imgSel);

clc;

%Set input folder
edge_inp = "edges";
gt_inp   = "gt";

%Set output folder
outp = "eval_out_p1d_manual";
if ~exist(outp,'dir')
    mkdir(outp);
end

%Command to set plot output folder
plotp = fullfile(outp, "plots");
if ~exist(plotp,'dir')
    mkdir(plotp);
end

%Command to set SE thresholds
se_thrs = [0.15 0.25 0.35];

fprintf("HW2 P1(d):\n");

%Command to run Bird evals
if imgSel == "ALL" || imgSel == "Bird"
    cfg.gt_mat = fullfile(gt_inp, "Bird_GT.mat");
    eval_img("Bird", cfg, edge_inp, se_thrs, outp, plotp);
end

%Command to run Deer evals
if imgSel == "ALL" || imgSel == "Deer"
    cfg.gt_mat = fullfile(gt_inp, "Deer_GT.mat");
    eval_img("Deer", cfg, edge_inp, se_thrs, outp, plotp);
end

fprintf("\nDone. Output File: %s\n", outp);
end

function eval_img(imgName, cfg, edge_inp, se_thrs, outp, plotp)

%Command to split GT
gt_outp = fullfile(outp, "tmp_gt");
gtFiles = make_gt_splits_once(cfg.gt_mat, gt_outp, imgName);
nGT = numel(gtFiles);

%Command to get Sobel maps
[sobel_files, sobelT] = pick_sobel(edge_inp, imgName);

%Command to get Canny maps
[canny_files, cannyT] = pick_canny(edge_inp, imgName);

%Command to grt SE map
seProb = fullfile(edge_inp, imgName + "_SE_prob.png");
if ~exist(seProb,'file')
    error("[%s] Missing SE prob map: %s", imgName, seProb);
end

%Part 1 CSV outputs + Part 2 CVS
sobel = fullfile(outp, sprintf("%s_Sobel_perGT.csv", imgName));
canny = fullfile(outp, sprintf("%s_Canny_perGT.csv", imgName));
struct = fullfile(outp, sprintf("%s_SE_perGT.csv", imgName));
summary = fullfile(outp, sprintf("%s_summary.csv", imgName));
sobelThrs = fullfile(outp, sprintf("%s_Sobel_thresholdTable.csv", imgName));
cannyThrs = fullfile(outp, sprintf("%s_Canny_thresholdTable.csv", imgName));
structThrs = fullfile(outp, sprintf("%s_SE_thresholdTable.csv", imgName));

%Command to run Sobel
[PRgt_s, thr_s] = eval_binary_set(imgName, "Sobel", sobel_files, gtFiles, "T", sobelT);
write_perGT(sobel, PRgt_s);
write_thres(sobelThrs, thr_s);

%Command to calc Sobel metrics
Ps = mean(PRgt_s(:,1));
Rs = mean(PRgt_s(:,2));
Fs = f_measure(Ps, Rs);

%Command to run Canny 
[PRgt_c, thr_c] = eval_binary_set(imgName, "Canny", canny_files, gtFiles, "pair", cannyT);
write_perGT(canny, PRgt_c);
write_thres(cannyThrs, thr_c);

%Command to calc Canny metrics
Pc = mean(PRgt_c(:,1));
Rc = mean(PRgt_c(:,2));
Fc = f_measure(Pc, Rc);

%Command to run SE
[PRgt_e, thr_e] = eval_probmap(imgName, "SE", seProb, se_thrs, gtFiles);
write_perGT(struct, PRgt_e);
write_thres(structThrs, thr_e);

%Command to calc SE metrics
Pe = mean(PRgt_e(:,1));
Re = mean(PRgt_e(:,2));
Fe = f_measure(Pe, Re);

%Write summary 
sumTbl = strings(3,4);
sumTbl(1,:) = ["Sobel", fmt(Ps), fmt(Rs), fmt(Fs)];
sumTbl(2,:) = ["Canny", fmt(Pc), fmt(Rc), fmt(Fc)];
sumTbl(3,:) = ["SE",    fmt(Pe), fmt(Re), fmt(Fe)];
write_sum(summary, sumTbl);
end

function [PR_gt, thr] = eval_binary_set(imgName, app, files, gtFiles, thrKind, thrLab)
K = numel(files);
nGT = numel(gtFiles);

P_all = zeros(K, nGT);
R_all = zeros(K, nGT);

%Command to loop over thresholds
for k = 1:K
    fprintf("[%s - %s] Thres Value %d / %d\n", imgName, app, k, K);
    Eb = read_raw_as_binary_uint8(files(k), 481, 321);

    %Command to loop over GT maps
    for i = 1:nGT
        fprintf("GT %d / %d\n", i, nGT);
        [P, R] = evalPR_exact(Eb, gtFiles(i));
        P_all(k,i) = P;
        R_all(k,i) = R;
    end
end

%Command to calc GT mean across thresholds
PR_gt = zeros(nGT,2);
for i = 1:nGT
    PR_gt(i,1) = mean(P_all(:,i));
    PR_gt(i,2) = mean(R_all(:,i));
end

%Command to group threshold data
thr.app = app;
thr.K = K;
thr.nGT = nGT;
thr.P_all = P_all;
thr.R_all = R_all;

%Command to build threshold labels
thr.thrLabel = strings(K,1);
for k = 1:K
    if thrKind == "T"
        thr.thrLabel(k) = "T" + string(thrLab(k));
    else
        thr.thrLabel(k) = string(thrLab(k));
    end
end

%Compute mean P R and F for each threshold
thr.meanP = mean(P_all, 2);
thr.meanR = mean(R_all, 2);
thr.F = arrayfun(@(a,b) f_measure(a,b), thr.meanP, thr.meanR);
end

function [PR_gt, thr] = eval_probmap(imgName, app, prob_file, thrs, gtFiles)
Eprob = read_png(prob_file);

K = numel(thrs);
nGT = numel(gtFiles);
P_all = zeros(K, nGT);
R_all = zeros(K, nGT);

%Command to loop over thresholds
for k = 1:K
    Eb = uint8(Eprob >= thrs(k));
    for i = 1:nGT
        [P, R] = evalPR_exact(Eb, gtFiles(i));
        P_all(k,i) = P;
        R_all(k,i) = R;
    end
end

%Command to compute per-GT mean across thresholds
PR_gt = zeros(nGT,2);
for i = 1:nGT
    PR_gt(i,1) = mean(P_all(:,i));
    PR_gt(i,2) = mean(R_all(:,i));
end

%Command to group threshold data
thr.app = app;
thr.K = K;
thr.nGT = nGT;
thr.P_all = P_all;
thr.R_all = R_all;

%Command to build threshold labels
thr.thrLabel = strings(K,1);
for k = 1:K
    thr.thrLabel(k) = sprintf("thr=%.2f", thrs(k));
end

%Compute mean P R and F for each threshold
thr.meanP = mean(P_all, 2);
thr.meanR = mean(R_all, 2);
thr.F = arrayfun(@(a,b) f_measure(a,b), thr.meanP, thr.meanR);
end

function [P, R] = evalPR_exact(Eb_uint8, gtFile)

Eb = (Eb_uint8 > 0);
S = load(char(gtFile));
gt1 = S.groundTruth{1};

%Command to get boundary map
if isfield(gt1, 'Boundaries')
    G = (gt1.Boundaries > 0);
else
    error("GT missing'Bounds' in %s", gtFile);
end

%Calc TP FP FN
TP = sum(Eb(:) & G(:));
FP = sum(Eb(:) & ~G(:));
FN = sum(~Eb(:) & G(:));

%Calc precision
if (TP + FP) == 0
    P = 0;
else
    P = TP / (TP + FP);
end

%Calc recall
if (TP + FN) == 0
    R = 0;
else
    R = TP / (TP + FN);
end
end

function [files, sobelT] = pick_sobel(folder, imgName)

%Command to list Sobel files
d = dir(fullfile(folder, imgName + "_edge_T*.raw"));
if isempty(d)
    error("[%s] No Sobel files found: %s", imgName, fullfile(folder, imgName + "_edge_T*.raw"));
end

%Command to sort files
names = sort(string({d.name}));
files = fullfile(folder, names);

%Command to parse threshold values
Tvals = zeros(numel(names),1);
for i = 1:numel(names)
    tok = regexp(names(i), "_T(\d+)\.raw$", "tokens", "once");
    if isempty(tok)
        Tvals(i) = i;
    else
        Tvals(i) = str2double(tok{1});
    end
end
sobelT = Tvals;
end

function [files, cannyT] = pick_canny(folder, imgName)

%Command to list Canny files
d = dir(fullfile(folder, imgName + "_canny_L*_H*.raw"));

%Command to sort files
names = sort(string({d.name}));
files = fullfile(folder, names);

%Command to parse low/high labels
cannyT = strings(numel(names),1);
for i = 1:numel(names)
    tok = regexp(names(i), "_L(\d+)_H(\d+)\.raw$", "tokens", "once");
    if isempty(tok)
        cannyT(i) = "pair" + string(i);
    else
        cannyT(i) = "L" + tok{1} + "_H" + tok{2};
    end
end
end

function gtFiles = make_gt_splits_once(gt_mat_path, tmp_dir, prefix)

if ~exist(tmp_dir, 'dir')
    mkdir(tmp_dir);
end

%Command to load GT mat
S = load(gt_mat_path);
if ~isfield(S,'groundTruth')
    error("GT mat missing 'groundTruth': %s", gt_mat_path);
end

%Command to split GT cell array
GTcell = S.groundTruth;
nGT = numel(GTcell);
gtFiles = strings(nGT,1);

for i = 1:nGT
    gtFiles(i) = fullfile(tmp_dir, sprintf("%s_GT%d.mat", prefix, i));
    if ~exist(gtFiles(i), 'file')
        groundTruth = {GTcell{i}}; 
        save(char(gtFiles(i)), 'groundTruth');
    end
end
end

function E = read_png(path)

%Command to read prob PNG 
I = imread(path);
if ndims(I) == 3
    I = rgb2gray(I);
end
E = double(I);
if max(E(:)) > 1
    E = E / 255.0;
end
E = max(0, min(1, E));
end

function Eb = read_raw_as_binary_uint8(path, width, height)

%Command to read RAW file
filesDir = fopen(path,'rb');
if filesDir < 0
    error("Cannot open inoput file %s", path);
end

I = fread(filesDir, width*height, 'uint8=>uint8');
fclose(filesDir);

I = reshape(I, [width height])';

%Convert RAW 
Eb = uint8(I < 128);
end

function write_perGT(csv_path, PR_gt)

%Command to write Per-GT CSV file
filesDir = fopen(csv_path, 'w');
fprintf(filesDir, "GT,MeanPrecision,MeanRecall\n");
for i = 1:size(PR_gt,1)
    fprintf(filesDir, "GT%d,%.6f,%.6f\n", i, PR_gt(i,1), PR_gt(i,2));
end
fclose(filesDir);
end

function write_sum(csv_path, summary)

%Write summary file
filesDir = fopen(csv_path, 'w');
fprintf(filesDir, "Method,MeanPrecision,MeanRecall,Fmeasure\n");
for i = 1:size(summary,1)
    fprintf(filesDir, "%s,%s,%s,%s\n", summary(i,1), summary(i,2), summary(i,3), summary(i,4));
end
fclose(filesDir);
end

function write_thres(csv_path, thr)

%Write threshold table
K = thr.K;
nGT = thr.nGT;

filesDir = fopen(csv_path, 'w');
fprintf(filesDir, "Threshold,MeanP,MeanR,F");
for i = 1:nGT
    fprintf(filesDir, ",GT%d_P,GT%d_R", i, i);
end
fprintf(filesDir, "\n");

for k = 1:K
    fprintf(filesDir, "%s,%.6f,%.6f,%.6f", thr.thrLabel(k), thr.meanP(k), thr.meanR(k), thr.F(k));
    for i = 1:nGT
        fprintf(filesDir, ",%.6f,%.6f", thr.P_all(k,i), thr.R_all(k,i));
    end
    fprintf(filesDir, "\n");
end

fclose(filesDir);
end

function F = f_measure(P,R)

%Command to compute F-measure
if (P + R) == 0
    F = 0;
else
    num = 2 * P * R;
    denom = P + R;
    F = num/ denom;
end
end

function s = fmt(x)

%Command to format numeric values
s = string(sprintf("%.6f", x));
end