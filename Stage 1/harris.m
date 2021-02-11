I = imread('image1.jpg');
I2 = rgb2gray(I);
corners = detectHarrisFeatures(I2);
points = detectSURFFeatures(I2);
[features, valid_points] = extractFeatures(I2, points);
imshow(I); hold on
plot(points);
