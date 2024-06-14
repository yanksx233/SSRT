function img_cropped = modcrop(img)
h = size(img, 1);
w = size(img, 2);

img_cropped = img(1:floor(h/8)*8, 1:floor(w/8)*8, :);
end

