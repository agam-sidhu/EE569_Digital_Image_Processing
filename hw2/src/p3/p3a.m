
% EE569 Homework #2
% Name: Agam Sidhu
% USC ID: 3027948957
% USC Email: agamsidh@usc.edu
% Submission Date: February 22, 2026
% Problem 3(a): Separable Error Diffusion Dithering for CMY

clear; clc; close all;

inp = "Flowers.raw";
width = 1280;
height = 853;
outp = "p3a_outputs/";

if ~exist(outp, 'dir')
    mkdir(outp);
end

inputRGB = readraw(inp, width, height, 3);

R = double(inputRGB(:,:,1));
G = double(inputRGB(:,:,2));
B = double(inputRGB(:,:,3));

C = 255 - R;
M = 255 - G;
Y = 255 - B;

stein = [0 0 0;
         0 0 7;
         3 5 1];
denomStein = 16;

Cb = errDiff(C, stein, denomStein);
Mb = errDiff(M, stein, denomStein);
Yb = errDiff(Y, stein, denomStein);
outR = uint8(255 - double(Cb));
outG = uint8(255 - double(Mb));
outB = uint8(255 - double(Yb));

outRGB = cat(3, outR, outG, outB);

writeraw(outp + "p3a_separable_CMY_FS.raw", outRGB);

function F = readraw(path, width, height, channels)
    fid = fopen(path, 'rb');
    assert(fid ~= -1, "Error: Cannot open input file.");

    F = fread(fid, width*height*channels, 'uint8=>double');
    fclose(fid);

    if channels == 1
        F = reshape(F, [width, height])';
    else
        
        F = reshape(F, [channels, width, height]);
        F = permute(F, [3 2 1]); % -> [height, width, channels]
    end

    F = uint8(F);
end

function writeraw(path, img)
    fid = fopen(path, 'wb');
    assert(fid ~= -1, "Error: Cannot write to output file.");

    if ndims(img) == 2
        fwrite(fid, img', 'uint8');
    else
        img = permute(img, [3 2 1]);         
        fwrite(fid, img(:), 'uint8');
    end

    fclose(fid);
end

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
