% EE569 Homework #2
% Name: Agam Sidhu
% USC ID: 3027948957
% USC Email: agamsidh@usc.edu
% Submission Date: February 22, 2026
% Problem 2(b): Error Diffusion Dithering for Grayscale
clear; clc; close all;

inp = "Reflection.raw";
outp = "p2b_outputs/";
width = 1280;
height = 852;

if ~exist(outp, 'dir')
    mkdir(outp);
end
input = readraw(inp, width, height);

%Steinberg Matrix + Run
stein = [0 0 0;
        0 0 7;
        3 5 1];
denomStein = 16;
steinOut = errDiff(input, stein, denomStein);
writeraw(outp + "p2b_FS.raw", steinOut);

%JJN Matrix + Run
jjn = [0 0 0 0 0;
         0 0 0 0 0;
         0 0 0 7 5;
         3 5 7 5 3;
         1 3 5 3 1];
denomJJN = 48;
jjnOut = errDiff(input, jjn, denomJJN);
writeraw(outp + "p2b_JJN.raw", jjnOut);

%Stucki Matrix + Run
stu = [0 0 0 0 0;
         0 0 0 0 0;
         0 0 0 8 4;
         2 4 8 4 2;
         1 2 4 2 1];
denomStu = 42;
stuOut = errDiff(input, stu, denomStu);
writeraw(outp + "p2b_Stucki.raw", stuOut);

%Read raw image function
function F = readraw(path, width, height)
    fid = fopen(path, 'rb');
    assert(fid ~= -1, "Error: Cannot open input file.");
    F = fread(fid, width*height, 'uint8=>double');
    fclose(fid);
    F = reshape(F, [width, height])';
end

%Write raw image function
function writeraw(path, img)
    fid = fopen(path, 'wb');
    assert(fid ~= -1, "Error: Cannot write to output file.");
    fwrite(fid, img', 'uint8');
    fclose(fid);
end

%Function tfor Error Diffusion
function G = errDiff(input, diffMat, den)
    input = double(input);
    [height, width] = size(input);
    G = zeros(height, width);
    halfK = floor(size(diffMat,1)/2);
    diffMat = diffMat / den;
    thresh = 128;

    for row = 1:height
        if mod(row,2) == 1
            colWidth = 1:width;
            dir = 1;
        else
            colWidth = width:-1:1;
            dir = -1;
        end

        for col = colWidth
            oldVal = input(row,col);

            if oldVal >= thresh
                newVal = 255;
            else
                newVal = 0;
            end
            G(row,col) = newVal;
            diff = oldVal - newVal;
            for kRow = 1:size(diffMat,1)
                for kCol = 1:size(diffMat,2)
                    wVal = diffMat(kRow,kCol);
                    if wVal == 0
                        continue;
                    end
                    nRow = row + (kRow - halfK - 1);
                    if dir == 1
                        nCol = col + (kCol - halfK - 1);
                    else
                        nCol = col - (kCol - halfK - 1);
                    end
                    if nRow >= 1 && nRow <= height && nCol >= 1 && nCol <= width
                        input(nRow,nCol) = input(nRow,nCol) + diff * wVal;
                        input(nRow,nCol) = min(255, max(0, input(nRow,nCol)));
                    end
                end
            end

        end
    end
    G = uint8(G);
end