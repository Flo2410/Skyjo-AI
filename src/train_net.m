% KI Aufgabe 
% Kienreich und Hye
clear all;

imds = imageDatastore("\Skyjo-AI\imgs\many\", 'IncludeSubfolders',true,'LabelSource','foldernames', "ReadFcn", @custom_read);

% zufällige Bilder aus Datenbank zeigen: nur für "Überblick"
figure(1);
perm = randperm(length(imds.Labels),20);
classes=length(unique(imds.Labels));

for i = 1:20
    subplot(4,5,i);
    p = perm(i);
    imshow(custom_read(imds.Files{p}));
    title(imds.Labels(p))
end

%% test
% for i = 1:length(imds.Files)
%     img = readimage(imds, i);
%     fprintf("%d %d %d - %s\n", size(img), imds.Files{i});
% end


%% Bilder aus "Datenbank" lesen
labelCount = countEachLabel(imds);
% img = readimage(imds,1);
% size(img)
numTrain = 150;        % Anzahl der Bilder die zum Trainieren verwendet werden
[imTrain,imVal] = splitEachLabel(imds,numTrain,'randomize'); % Daten teilen

fprintf("Train Count: %d\n", length(imTrain.Files));
fprintf("Val Count: %d\n", length(imVal.Files));

%% Layers
filter_size = 6;
layers = [
    imageInputLayer([500 280 3])

    convolution2dLayer(filter_size, 8, 'Padding', "same")
    batchNormalizationLayer()
    reluLayer()
    maxPooling2dLayer([2, 2],'Stride',2)
%     dropoutLayer(0.2)

    convolution2dLayer(filter_size,16,'Padding',"same")
    batchNormalizationLayer()
    reluLayer()
    maxPooling2dLayer([2, 2],'Stride',2)
%     dropoutLayer(0.2)
    
    convolution2dLayer(filter_size,32,'Padding',"same")
    batchNormalizationLayer()
    reluLayer()
    maxPooling2dLayer([2, 2],'Stride',2)

    convolution2dLayer(filter_size,64,'Padding',"same")
    batchNormalizationLayer()
    reluLayer()
   % maxPooling2dLayer([2, 2],'Stride',2)

    fullyConnectedLayer(classes)
    softmaxLayer()
    classificationLayer()
    ];

% Plot von Layers:
% figure(2);
% plot(layerGraph(layers));

% Optionen für Training
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imVal, ...
    'ValidationFrequency',15, ...
    'Verbose',false, ...
    'MiniBatchSize', 64, ...
    'Plots','training-progress');

net = trainNetwork(imTrain,layers,options);



%% Bilder "testen"
Labels=categories(unique(imds.Labels));  % "Bezeichnung ermitteln"
perm = randperm(length(imds.Labels),20); % Zufällige Auswahl von Bildern
figure("Name","Test Predictions"); clf;
for i = 1:20
    subplot(4,5,i);
    X=custom_read(imds.Files{perm(i)});
    imshow(X);
    y = predict(net, X);
    [~, ind] = max(y);
    title(Labels{ind});
end

save("net", "net", "Labels");
