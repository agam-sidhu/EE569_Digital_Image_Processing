clc; clear; close all;

% ============================================================
% EE569 HW2 - Problem 1(d): Quantitative Evaluation
% Uses Ground Truth (.mat) and edge maps (Sobel/Canny/SE)
% Folder structure (relative to this script):
%   p1d/
%     p1d_eval_edges.m
%     edges/  (generated edge maps)
%     gt/     (ground truth .mat files)
% ============================================================

% ---- Paths (relative to p1d/) ----
gtDir   = 'gt';
edgeDir = 'edges';

% ---- Images + corresponding GT files (your exact names) ----
images = { ...
  struct('name','Bird', 'gtMat', fullfile(gtDir,'Bird_GT.mat')), ...
  struct('name','Deer', 'gtMat', fullfile(gtDir,'Deer_GT.mat')) ...
};

% ---- Detectors + your exact edge filenames ----
% For Sobel and Canny you currently have binary edge maps (PNG).
% For SE you have a probability map (PNG) which supports threshold sweep.
detectors = { ...
  struct('label','Sobel', 'filePattern','%s_edge_T25.png'), ...
  struct('label','Canny', 'filePattern','%s_canny_L60_H180.png'), ...
  struct('label','SE',    'filePattern','%s_SE_prob.png') ...
};

% ---- Threshold sweep range ----
thrs = 0.00:0.01:1.00;

% ---- Matching tolerance for correspondPixels (BSDS-style) ----
% Standard approach uses a distance proportional to image diagonal.
tolFrac = 0.0075;

% ---- Output folder for plots and CSV tables ----
outDir = fullfile('eval_out');
if ~exist(outDir,'dir'), mkdir(outDir); end

% ============================================================
% MAIN LOOP
% ============================================================
for d = 1:numel(detectors)
  det = detectors{d};

  fprintf('\n=========================================\n');
  fprintf('Detector: %s\n', det.label);
  fprintf('=========================================\n');

  for im = 1:numel(images)
    info = images{im};

    % ----- Load GT -----
    gtData = load(info.gtMat);
    groundTruth = extractGroundTruth(gtData); % 1x5 cell expected

    % ----- Load edge "strength" map E in [0,1] -----
    edgePath = fullfile(edgeDir, sprintf(det.filePattern, info.name));
    E = loadEdgeStrength(edgePath);

    % ==========================================================
    % (1) Precision/Recall table for each GT at ONE chosen threshold
    % For SE, pick a fixed example threshold (matches your 1c writeup)
    % For Sobel/Canny, the PNG is likely already binary, but thresholding still works.
    % ==========================================================
    Tfixed = 0.20;
    Eb = E >= Tfixed;

    [P_each, R_each, Pmean, Rmean, F] = evalPRF_forAllGT(Eb, groundTruth, tolFrac);

    fprintf('\nImage: %s | Fixed threshold T = %.2f\n', info.name, Tfixed);
    fprintf('GT#   Precision   Recall\n');
    for k = 1:numel(P_each)
      fprintf('%d     %0.4f     %0.4f\n', k, P_each(k), R_each(k));
    end
    fprintf('Mean  %0.4f     %0.4f\n', Pmean, Rmean);
    fprintf('F     %0.4f\n', F);

    % Save table to CSV
    csvPath = fullfile(outDir, sprintf('%s_%s_PR_table_T%.2f.csv', info.name, det.label, Tfixed));
    writePRTableCSV(csvPath, P_each, R_each, Pmean, Rmean, F);

    % ==========================================================
    % (2) F-measure vs Threshold curve
    % This is most meaningful for SE (probability map).
    % For binary Sobel/Canny maps, curve may be mostly flat.
    % ==========================================================
    Fcurve = zeros(size(thrs));
    Pcurve = zeros(size(thrs));
    Rcurve = zeros(size(thrs));

    for t = 1:numel(thrs)
      Eb_t = E >= thrs(t);
      [~, ~, Pm, Rm, Ft] = evalPRF_forAllGT(Eb_t, groundTruth, tolFrac);
      Pcurve(t) = Pm;
      Rcurve(t) = Rm;
      Fcurve(t) = Ft;
    end

    [Fbest, idxBest] = max(Fcurve);
    thrBest = thrs(idxBest);

    fprintf('Threshold sweep (%s): best F = %0.4f at T = %0.2f\n', info.name, Fbest, thrBest);

    % Plot F vs threshold
    fig = figure('Visible','on');
    plot(thrs, Fcurve, 'LineWidth', 1.5);
    xlabel('Threshold T');
    ylabel('F-measure');
    title(sprintf('%s: F vs Threshold (%s)', det.label, info.name));
    grid on;

    outPlot = fullfile(outDir, sprintf('%s_%s_Fcurve.png', info.name, det.label));
    saveas(fig, outPlot);

    % Also save curve values
    curveCSV = fullfile(outDir, sprintf('%s_%s_Fcurve.csv', info.name, det.label));
    writeCurveCSV(curveCSV, thrs, Pcurve, Rcurve, Fcurve, Fbest, thrBest);

    close(fig);
  end
end

fprintf('\nDone. Outputs saved in: %s\n', outDir);

% ============================================================
% HELPER FUNCTIONS
% ============================================================

function groundTruth = extractGroundTruth(gtData)
  % Standard BSDS format: gtData.groundTruth (cell array)
  if isfield(gtData, 'groundTruth')
    groundTruth = gtData.groundTruth;
    return;
  end

  % Sometimes different field name but same structure
  fn = fieldnames(gtData);
  for i = 1:numel(fn)
    val = gtData.(fn{i});
    if iscell(val) && ~isempty(val) && isstruct(val{1}) && isfield(val{1}, 'Boundaries')
      groundTruth = val;
      return;
    end
  end
  error('Could not find groundTruth cell array in the GT .mat file.');
end

function E = loadEdgeStrength(path)
  if ~exist(path,'file')
    error('Edge file not found: %s', path);
  end

  img = imread(path);
  if ndims(img) == 3
    img = img(:,:,1);
  end
  E = double(img) / 255.0;

  % Clamp
  E(E < 0) = 0;
  E(E > 1) = 1;
end

function [P_each, R_each, Pmean, Rmean, F] = evalPRF_forAllGT(Eb, groundTruth, tolFrac)
  nGT = numel(groundTruth);
  P_each = zeros(nGT,1);
  R_each = zeros(nGT,1);

  for i = 1:nGT
    G = logical(groundTruth{i}.Boundaries);
    [P_each(i), R_each(i)] = evalPR_oneGT(Eb, G, tolFrac);
  end

  Pmean = mean(P_each);
  Rmean = mean(R_each);
  F = (2 * Pmean * Rmean) / (Pmean + Rmean + eps);
end

function [P, R] = evalPR_oneGT(Eb, G, tolFrac)
  [h,w] = size(G);
  maxDist = tolFrac * sqrt(double(h*h + w*w));

  % Requires pdollar/edges on path
  [m1, m2] = correspondPixels(Eb, G, maxDist);

  TP = sum(m1(:));
  FP = sum(Eb(:)) - TP;
  FN = sum(G(:)) - sum(m2(:));

  P = TP / (TP + FP + eps);
  R = TP / (TP + FN + eps);
end

function writePRTableCSV(csvPath, P_each, R_each, Pmean, Rmean, F)
  fid = fopen(csvPath,'w');
  fprintf(fid, 'GT,Precision,Recall\n');
  for k = 1:numel(P_each)
    fprintf(fid, '%d,%0.6f,%0.6f\n', k, P_each(k), R_each(k));
  end
  fprintf(fid, 'Mean,%0.6f,%0.6f\n', Pmean, Rmean);
  fprintf(fid, 'F-measure,%0.6f,\n', F);
  fclose(fid);
end

function writeCurveCSV(csvPath, thrs, Pcurve, Rcurve, Fcurve, Fbest, thrBest)
  fid = fopen(csvPath,'w');
  fprintf(fid, 'T,MeanPrecision,MeanRecall,MeanF\n');
  for i = 1:numel(thrs)
    fprintf(fid, '%0.2f,%0.6f,%0.6f,%0.6f\n', thrs(i), Pcurve(i), Rcurve(i), Fcurve(i));
  end
  fprintf(fid, 'Best,,,\n');
  fprintf(fid, 'BestT,%0.2f,,\n', thrBest);
  fprintf(fid, 'BestF,,,%0.6f\n', Fbest);
  fclose(fid);
end