n_images = 19;
img = imread(strcat('code/images/image1.jpg'));
sze = size(img);
%index = 8;
for i=1:n_images
    %if(i ~= index)
        [d, f] = readBinaryDescriptors(strcat('dcsift',num2str(i)));
        if(i==1)
            feat_set = zeros(size(d,1),size(d,2),n_images);
            %X = zeros(size(d,1),size(d,2));
            X = d;
        end
        if(i>1)
            X = vertcat(X,d);
        end
        feat_set(:,:,i) = d;
    %end
end

n_codebooks = 100;
% [C, idx] = kmeans(X',100); 
% save('codebook_8.mat','C');


linearInd = sub2ind([sze(1) sze(2)], f(:,2), f(:,1));

load codebook_8.mat

knn = 5;
label  = zeros(1,1);
llc_labelled_final = [];
R1 = [];
for i=1:n_images
    img = imread(strcat('code/images/image',num2str(i),'.jpg'));
    if(size(img,1) > size(img,2))
        img = imrotate(img,-90);
    end
    
    img_groundtruth = imread(strcat('code/ground_truth/ground_truth',num2str(i),'.jpg'));
    if(size(img_groundtruth,1) > size(img_groundtruth,2))
        img_groundtruth = imrotate(img_groundtruth,-90);
    end
    [Coeff] = LLC_coding_appr(C', feat_set(:,:,i), 5, 1e-4);
    

    regionSize = 40 ;
    regularizer = 0.5 ;
    superPixels = vl_slic(single(img), regionSize, regularizer) ;
    string = strcat('superPixels_40_8',num2str(i),'.mat');
    save(string,'superPixels');
    perim = true(size(img,1), size(img,2));

    for k = 0:size (unique (superPixels), 1)-1
        I = find (superPixels == k);    %linear indices of the point in kth superpixel
        Lia = ismember(linearInd,I);    %logical 1 where indices of superpixel is found
        count2 = sum(Lia(:) == 1);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        regionK = superPixels == k;
        perimK = bwperim(regionK, 8);
        perim(perimK) = false;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        segk_codes = Coeff(Lia,:);      %codes of sampled points inside the superpixel
        R = img_groundtruth(I);
        R1 = R > 0;
        count1 = sum(R1(:) == 1);
%         if(count1 > 0)
%             count1
%             k
%         end
        count = size(segk_codes,1);
        if(count > 0)
            fract_object_area = double(count1)/double(size(I,1))
            if(fract_object_area > 0.98)
               label(1) = 1;
            else if(fract_object_area == 0)
                label(1) = 0;
                end
            end
            llc = maxpooling(segk_codes)/norm(max(segk_codes));
            llc_labeled = [llc label i k];
            llc_labelled_final = [llc_labelled_final;llc_labeled];
        end
    end
    perim = uint8(cat(3,perim,perim,perim));
    finalImage = img .* perim;
    imshow(finalImage);
    %imshow(img_groundtruth);
 end
save('data_40_8.mat','llc_labelled_final');
