function plot_p1d(whichImage)
% EE569 Homework #2
% Name: Agam Sidhu
% USC ID: 3027948957
% USC Email: agamsidh@usc.edu
% Submission Date: February 22, 2026
% Problem 1(d): Evaluation of Edge Detectors - Plotting F vs Threshold

if nargin < 1, whichImage = "ALL"; end
whichImage = string(whichImage);
outp  = "eval_out_p1d_manual";
plotp = fullfile(outp, "plots");
if ~exist(plotp,'dir'), mkdir(plotp); end

if whichImage == "ALL" || whichImage == "Bird"
    make_one("Bird", outp, plotp);
end
if whichImage == "ALL" || whichImage == "Deer"
    make_one("Deer", outp, plotp);
end
fprintf("Done. Plots saved to: %s\n", plotp);
end

function make_one(imgName, outp, plotp)
methods = ["Sobel","Canny","SE"];

for m = methods

    csvPath = fullfile(outp, sprintf("%s_%s_thresholdTable.csv", imgName, m));
    if ~exist(csvPath,'file')
        warning("Missing CSV: %s", csvPath);
        continue;
    end

    Ttbl = readtable(csvPath);

    Tvals  = Ttbl.Threshold;
    Fvals  = Ttbl.F;

    % Convert threshold column to string labels
    Tlabels = string(Tvals);

    fig = figure('Visible','on');
    plot(1:numel(Fvals), Fvals, '-o', 'LineWidth', 2);
    grid on;

    xlabel('Threshold');
    ylabel('F-measure (avg over GTs)');
    title(sprintf('%s - %s: F vs Threshold', imgName, m));

    ax = gca;
    ax.XTick = 1:numel(Fvals);
    ax.XTickLabel = Tlabels;
    ax.XTickLabelRotation = 25;

    drawnow;

    outFile = fullfile(plotp, sprintf("%s_%s_F_vs_threshold.png", imgName, m));
    exportgraphics(fig, outFile);

    close(fig);
end
end