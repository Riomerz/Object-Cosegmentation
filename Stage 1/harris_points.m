% Harris detector
% The code calculates
% the Harris Feature Points(FP) 
% 
% When u execute the code, the test image file opened
% and u have to select by the mouse the region where u
% want to find the Harris points, 
% then the code will print out and display the feature
% points in the selected region.
% You can select the number of FPs by changing the variables 
% max_N & min_N
% A. Ganoun

frame=imread('image1.jpg');

I =double(frame);
%****************************
imshow(frame);
% k = waitforbuttonpress;
% point1 = get(gca,'CurrentPoint');  %button down detected
% rectregion = rbbox;  %%%return figure units
% point2 = get(gca,'CurrentPoint');%%%%button up detected
% point1 = point1(1,1:2); %%% extract col/row min and maxs
% point2 = point2(1,1:2);
% lowerleft = min(point1, point2);
% upperright = max(point1, point2); 
% ymin = round(lowerleft(1)); %%% arrondissement aux nombrs les plus proches
% ymax = round(upperright(1));
% xmin = round(lowerleft(2));
% xmax = round(upperright(2));
%***********************************
% Aj=1;
% cmin=xmin-Aj; cmax=xmax+Aj; rmin=ymin-Aj; rmax=ymax+Aj;
min_N=80;max_N=150;
%%%%%%%%%%%%%%Intrest Points %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma=2; Thrshold=20; r=6; disp=1;
dx = [-1 0 1; -1 0 1; -1 0 1]; % The Mask 
    dy = dx';
    %%%%%% 
    [m,n,o] = size(I);
    m = fix(m);
    n = fix(n);
    Ix = conv2(I(1:m,1:n), dx, 'same');   
    Iy = conv2(I(1:m,1:n), dy, 'same');
    g = fspecial('gaussian',max(1,fix(6*sigma)), sigma); %%%%%% Gaussien Filter
    
    %%%%% 
    Ix2 = conv2(Ix.^2, g, 'same');  
    Iy2 = conv2(Iy.^2, g, 'same');
    Ixy = conv2(Ix.*Iy, g,'same');
    %%%%%%%%%%%%%%
    k = 0.04;
    R1 = (Ix2.*Iy2 - Ixy.^2) - k*(Ix2 + Iy2).^2;
    
    f = (1000/max(max(R1))).*R1;
    %fprintf('%d ', (100/max(max(R1))).*R1);
    R11=(1000/max(max(R1))).*R1;
    R=R11;
    %fprintf('%d ', R);
    ma=max(max(R));
    sze = 2*r+1; 
    MX = ordfilt2(R,sze^2,ones(sze));
    R11 = (R==MX)&(R>Thrshold); 
    count=sum(sum(R11(2:size(R11,1)-2,2:size(R11,2)-2)));
    
    
    loop=0;
    while (((count<min_N)|(count>max_N))&(loop<30))
        if count>max_N
            Thrshold=Thrshold*1.5;
        elseif count < min_N
            Thrshold=Thrshold*0.5;
        end
        
        R11 = (R==MX)&(R>Thrshold); 
        count=sum(sum(R11(2:size(R11,1)-2,2:size(R11,2)-2)));
        loop=loop+1;
    end
    
    
	R=R*0;
    R(2:size(R11,1)-2,2:size(R11,2)-2)=R11(2:size(R11,1)-2,2:size(R11,2)-2);
	[r1,c1] = find(R);
    PIP=[r1,c1]%% IP 
   

   %%%%%%%%%%%%%%%%%%%% Display
   
   Size_PI=size(PIP,1);
   for r=1: Size_PI
   I(PIP(r,1)-1:PIP(r,1)+1,PIP(r,2)-1)=255;
   I(PIP(r,1)-1:PIP(r,1)+1,PIP(r,2)+1)=255;
   I(PIP(r,1)-1,PIP(r,2)-1:PIP(r,2)+1)=255;
   I(PIP(r,1)+1,PIP(r,2)-1:PIP(r,2)+1)=255;
   
   end
   
   imshow(uint8(I))