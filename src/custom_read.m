function data = custom_read(filename)
    img = imread(filename);
    %img = rgb2gray(img);
    %img = imgaussfilt(img,4);
    %data = imbinarize(img);
    %img = img > 190;
    img = imrotate(img, randi([-30 30],1,1));
    data = imresize(img, [500, 280]);
end

