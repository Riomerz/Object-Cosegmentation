img=imread('image1.jpg');
img_height  = size(img,1);
img_width   = size(img,2);
x = floor((img_width-7)/6) - 1;
y = floor((img_height-7)/6) - 1;
n_images = 19;


for k=1:n_images
    img=imread(strcat('image',num2str(k),'.jpg'));
    points = zeros(x*y,2);
    lamda = 1;

    for j=0:y
        for i=0:x
            if(mod(j,2) == 0 && 10 +(6*i) <= img_width-7)
                points(lamda,1) = 10 +(6*i);
                points(lamda,2) = 7 + (6*j);
                lamda = lamda + 1;
            else if(mod(j,2) == 1 && 7 +(6*i) <= img_width-7)
                points(lamda,1) = 7 +(6*i);
                points(lamda,2) = 7 + (6*j);
                lamda = lamda + 1;
                end
            end

        end
    end
    
    if(k == 1)
        X = zeros(size(points,1),2,n_images);
        img_d = zeros(128*3,size(points,1),n_images);
    end
    
    %in dense sampling the coordinate points would remain the same for all images so no
    %need to keep a record of points for all images
    
    X(:,:,k) = points;
    
    o1 = (img(:,:,1) - img(:,:,2))./sqrt(2);
    o2 = (img(:,:,1) + img(:,:,2) - 2.*img(:,:,3))./sqrt(6);
    o3 = (img(:,:,1) + img(:,:,2) + img(:,:,3))./sqrt(3);


    %Opponent SIFT
    H1 = single(o1);
    H2 = single(o2);
    H3 = single(o3);

    ori = zeros(size(points,1),1);
    scale = zeros(size(points,1),1);
    for i=1:size(points,1)
        scale(i) = 1.2;
    end
    
    fc =horzcat(points,scale,ori) ;

    [f,d1] = vl_sift(H1,'frames',transpose(fc)) ;
    [f,d2] = vl_sift(H2,'frames',transpose(fc)) ;
    [f,d3] = vl_sift(H3,'frames',transpose(fc)) ;

    img_d(:,:,k) = vertcat(d1,d2,d3);

    imshow(img);
    hold on
    plot(points(:,1),points(:,2),'r.');
    hold off
end