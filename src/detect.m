% KI Aufgabe 
% Kienreich und Hye
clear all;
load("detector.mat")

img = imread("test2.jpg");

figure(1)
imshow(img)

[bbox, score, label] = detect(detector, img, 'MiniBatchSize', 64);
[score, idx] = max(score);

bbox = bbox(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);

figure(2)
imshow(detectedImg)
