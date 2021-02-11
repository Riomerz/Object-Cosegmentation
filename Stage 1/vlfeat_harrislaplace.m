imgs=imread('image1.jpg');
% gray_img = rgb2gray(imgs)
% I = im2single(gray_img);
%a = [0.06,0.63,0.27;0.3,0.04,-0.35;0.34,-0.6,0.17];
img = double(imgs);


o1 = (img(:,:,1) - img(:,:,2))./sqrt(2);
o2 = (img(:,:,1) + img(:,:,2) - 2.*img(:,:,3))./sqrt(6);
o3 = (img(:,:,1) + img(:,:,2) + img(:,:,3))./sqrt(3);
H = o1./o3;
H = single(H);

%frames = vl_covdet(H, 'verbose') ;
frames = vl_covdet(H, 'method', 'HarrisLaplace') ;
T = frames(1:2) ;
A = reshape(frames(3:6),2,2) ;
hold on ;
imshow(imgs);
vl_plotframe(frames) ;