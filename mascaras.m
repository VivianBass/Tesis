clear all; 
close all;

imagenes=dir('path/*.png');
im='path/';
[vect,idx]=sort([imagenes.datenum]);
imagenes=imagenes(idx);

for i=1:1:length(imagenes)
    indice=imagenes(i).name;
    all=[im indice];
    disp(indice);
    
    img = rgb2gray(imread(all));
    
    figure;
    subplot(1,2,1);
    imshow(img);
    
    coord1=[]; coord2=[];
    
    [i,j]=size(img); a=i/2; b=j/2;
    
    for x=1:a
        for y=1:b
            if img(x,y)>0 && isempty(coord1)
                coord1=[x y];
            end
        end
    end
    
    for x=i:-1:1
        for y=j:-1:1
            if img(x,y)>0 && isempty(coord2)
                coord2=[x y];
            end
        end
    end
    
    x=coord2(1)-coord1(1); y=coord2(2)-coord1(2);
    
    new=zeros(x,y);
    
    for i=1:x
        for j=1:y
            new(i,j)=img(i+coord1(1),j+coord1(2));
        end
    end
    
    x2=coord2(1)-coord1(1)-16; y2=coord2(2)-coord1(2)-14;
    newsinBorde=zeros(x2,y2);
    
    for i=1:x2
        for j=1:y2
            newsinBorde(i,j)=new(i+8,j+7);
        end
    end
       
    I2 = imfill(newsinBorde); 
    x3=x2+8; y3=y2+7;
    new(9:x3,8:y3)=I2;     
    new(9:48,8:124)=0; 
    [in,jn]=size(new);
    new(1:in,1:10)=0; 
    new(1:10,1:jn)=0; 
    new(in-10:in,1:jn)=0;
    new(1:in,jn-10:jn)=0; 
    new = imbinarize(new,9); 
    subplot(1,2,2)
    imshow(new,[]);
    %imwrite(new,all);
end