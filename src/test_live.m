load("net_final_v2_color.mat");

vid = videoinput("winvideo");
preview(vid);

%%
sum = 0;
while 1
    input("Press Button!");

    color_frame = getsnapshot(vid);

    frame = rgb2gray(color_frame);
    frame = imgaussfilt(frame,3);

    frame = frame > 200;

    [row,col] = ind2sub(size(frame), find(frame));
    
    row_min = min(row) - 20;
    row_max = max(row) + 20;
    col_min = min(col) - 20;
    col_max = max(col) + 20;

    if row_min < 0
        row_min = 1;
    end
    if row_max < 0
        row_max = 1;
    end
    if col_min < 0
        col_min = 1;
    end
    if col_max < 0
        col_max = 1;
    end

     if row_min > size(frame, 1)
        row_min = size(frame, 1) - 1 ;
    end
    if row_max  > size(frame, 1)
        row_max = size(frame, 1) - 1;
    end
    if col_min > size(frame, 2)
        col_min = size(frame, 2) - 1;
    end
    if col_max > size(frame, 2)
        col_max = size(frame, 2) - 1;
    end
    
    new_frame = frame(row_min:row_max, col_min:col_max, :);
    new_color_frame = color_frame(row_min:row_max, col_min:col_max, :);
    
    figure(2)
    montage({new_color_frame,new_frame});
    
    new_frame = imresize(new_frame, [500, 280]);
    new_color_frame = imresize(new_color_frame, [500, 280]);

    y = predict(net, new_color_frame);
    [~, ind] = max(y);
    int = get_int(Labels{ind});
    title(sprintf("%s - %d", string(Labels{ind}), int));
    sum = sum + int
end

function int = get_int(str)
    switch str
        case "m2"
            int = -2;
        case "m1"
            int = -1;
        case "zero"
            int = 0;
        case "one"
            int = 1;
        case "two"
            int = 2;
        case "three"
            int = 3;
        case "four"
            int = 4;
        case "five"
            int = 5;
        case "six"
            int = 6;
        case "seven"
            int = 7;
        case "eight"
            int = 8;
        case "nine"
            int = 9;
        case "ten"
            int = 10;
        case "eleven"
            int = 11;
        case "twelth"
            int = 12;
        case "back"
            int = 0;
    end
end
