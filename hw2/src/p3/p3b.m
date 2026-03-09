% EE569 Homework #2
% Name: Agam Sidhu
% USC ID: 3027948957
% USC Email: agamsidh@usc.edu
% Submission Date: February 22, 2026
% Problem 3(b): MBVQ Error Diffusion Dithering for RGB
clear; clc; close all;

inp = "Flowers.raw";
outp = "p3b_outputs/";
width = 1280;
height = 853;

if ~exist(outp, 'dir')
    mkdir(outp);
end

inputRGB = readraw(inp, width, height, 3);

stein = [0 0 0;
         0 0 7;
         3 5 1];
denomStein = 16;

mbvqOut = errDiff_MBVQ(inputRGB, stein, denomStein);
writeraw(outp + "p3b_MBVQ_FS.raw", mbvqOut);

function F = readraw(path, width, height, channels)
    fid = fopen(path, 'rb');
    assert(fid ~= -1, "Error: Cannot open input file.");
    F = fread(fid, width*height*channels, 'uint8=>double');
    fclose(fid);

    if channels == 1
        F = reshape(F, [width, height])';
    else
        F = reshape(F, [channels, width, height]);
        F = permute(F, [3 2 1]);  % (H,W,C)
    end

    F = uint8(F);
end

function writeraw(path, img)
    fid = fopen(path, 'wb');
    assert(fid ~= -1, "Error: Cannot write to output file.");

    if ndims(img) == 2
        fwrite(fid, img', 'uint8');
    else
        img = permute(img, [3 2 1]); % (C,W,H)
        fwrite(fid, img(:), 'uint8');
    end
    fclose(fid);
end

function outRGB = errDiff_MBVQ(inputRGB, diffMat, den)
    inputRGB = double(inputRGB);
    [H, W, ~] = size(inputRGB);

    bufR = inputRGB(:,:,1);
    bufG = inputRGB(:,:,2);
    bufB = inputRGB(:,:,3);
    outRGB = zeros(H, W, 3);

    halfK = floor(size(diffMat,1)/2);
    diffMat = diffMat / den;

    for row = 1:H
        if mod(row,2) == 1
            colRange = 1:W;
            dir = 1;      
        else
            colRange = W:-1:1;
            dir = -1;    
        end

        for col = colRange
            r0 = inputRGB(row,col,1);
            g0 = inputRGB(row,col,2);
            b0 = inputRGB(row,col,3);
            mbvq = getMBVQ(r0, g0, b0);  

            Rc = min(255, max(0, bufR(row,col)));
            Gc = min(255, max(0, bufG(row,col)));
            Bc = min(255, max(0, bufB(row,col)));

            vName = getNearestVertex(mbvq, Rc/255, Gc/255, Bc/255); 
            V = vertexToRGB(vName);  

            outRGB(row,col,1) = V(1);
            outRGB(row,col,2) = V(2);
            outRGB(row,col,3) = V(3);

            errR = Rc - V(1);
            errG = Gc - V(2);
            errB = Bc - V(3);
            for kRow = 1:size(diffMat,1)
                for kCol = 1:size(diffMat,2)
                    w = diffMat(kRow,kCol);
                    if w == 0, continue; end

                    nRow = row + (kRow - halfK - 1);
                    if dir == 1
                        nCol = col + (kCol - halfK - 1);
                    else
                        nCol = col - (kCol - halfK - 1);
                    end

                    if nRow >= 1 && nRow <= H && nCol >= 1 && nCol <= W
                        bufR(nRow,nCol) = min(255, max(0, bufR(nRow,nCol) + errR*w));
                        bufG(nRow,nCol) = min(255, max(0, bufG(nRow,nCol) + errG*w));
                        bufB(nRow,nCol) = min(255, max(0, bufB(nRow,nCol) + errB*w));
                    end
                end
            end

        end
    end
    outRGB = uint8(outRGB);
end

function mbvq = getMBVQ(r, g, b)
    if (r + g) > 255
        if (g + b) > 255
            if (r + g + b) > 510
                mbvq = "CMYW";
            else
                mbvq = "MYGC";
            end
        else
            mbvq = "RGMY";
        end
    else
        if (g + b) <= 255
            if (r + g + b) <= 255
                mbvq = "KRGB";
            else
                mbvq = "RGBM";
            end
        else
            mbvq = "CMGB";
        end
    end
end

function V = vertexToRGB(vName) 
    switch char(vName)
        case 'black',   V = [0 0 0];
        case 'white',   V = [255 255 255];
        case 'red',     V = [255 0 0];
        case 'green',   V = [0 255 0];
        case 'blue',    V = [0 0 255];
        case 'cyan',    V = [0 255 255];
        case 'magenta', V = [255 0 255];
        case 'yellow',  V = [255 255 0];
        otherwise,      V = [0 0 0]; % fallback
    end
end