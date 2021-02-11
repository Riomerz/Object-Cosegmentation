I = imread('image1.jpg');
%img = rgb2gray(I);
img = double(I);
o1 = (img(:,:,1) - img(:,:,2))./sqrt(2);
o2 = (img(:,:,1) + img(:,:,2) - 2.*img(:,:,3))./sqrt(6);
o3 = (img(:,:,1) + img(:,:,2) + img(:,:,3))./sqrt(3);
H = o1;

points = kp_harrislaplace(H);
r = points(:,1);
c = points(:,2);
scale = points(:,3);
n = size(points,1);
ori = zeros(n,1);

% for i=1:n
%    ori(i) = atan((img(x(i),y(i)+1) - img(x(i),y(i)-1)) / (img(x(i)+1,y(i)) - img(x(i)-1,y(i))));
% end
fc =horzcat(points,ori) ;
% img = single(img);
H = single(H);
[f,d] = vl_sift(H,'frames',transpose(fc),'orientations') ;
imshow(I); hold on
scatter(c,r);
