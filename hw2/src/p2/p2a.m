% EE569 Homework #2
% Name: Agam Sidhu
% USC ID: 3027948957
% USC Email: agamsidh@usc.edu
% Submission Date: February 22, 2026
% Problem 2(a): Thresholding and Dithering for Grayscale

clear; clc; close all;
inp = "Reflection.raw";
outp = "outputs/";
width = 1280;
height = 852;

if ~exist(outp, 'dir')
    mkdir(outp);
end

F = readraw(inp, width, height);

% Command to run Fixed Threshold
fixedT = 128;
fix_G  = uint8(255) * uint8(F >= fixedT);
writeraw(outp + "p2a_fixed.raw", fix_G);

% Command to run Random Threshold
randomT = 0;   
rng(randomT);
R = randi([0 255], size(F));
rand_G = uint8(255) * uint8(F >= R);
writeraw(outp + "p2a_random.raw", rand_G);

%Command to run Dither
nVal = [2 8 32];
for k = 1:length(nVal)

    N = nVal(k);
    I = buildBayer(N);
    T = buildThreshold(I);
    dit_G = dither(F, T);

    writeraw(outp + sprintf("p2a_I%d.raw", N), dit_G);
    imwrite(uint8(T), outp + sprintf("T%d.png", N));

end

%Read raw image function
function F = readraw(path, width, height)
    fid = fopen(path, 'rb');
    assert(fid ~= -1, "Error: Cannot open input file.");
    F = fread(fid, width*height, 'uint8=>uint8');
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

%Build bayer matrix function
function I = buildBayer(N)
    I = uint16([1 2;
                3 0]);   
    curVal = 2;
    while curVal < N
        I = [4*I + 1, 4*I + 2;
             4*I + 3, 4*I];
        curVal = curVal * 2;
    end
end

% Build Threshold Matrix Functiomn
function T = buildThreshold(I)
    N = size(I,1);
    T = (double(I) + 0.5) / (N*N) * 255;
end

%Dithering Function
function G = dither(F, T)
    [height, width] = size(F);
    N = size(T,1);
    threshMap = zeros(height, width);
    for r = 1:height
        for c = 1:width
            row = mod(r-1, N) + 1;
            col = mod(c-1, N) + 1;
            threshMap(r,c) = T(row, col);
        end
    end
    G = uint8(255) * uint8(double(F) > threshMap);
end