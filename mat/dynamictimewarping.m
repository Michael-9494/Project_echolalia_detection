function [temp]=dynamictimewarping(S1,S2)
%If S1 is the query speech
% signal and S2 is in the database
%after endpoint detection
figure
plot(S1)
hold on
plot(S2,'r')
title('Query and the reference speech signal after endpoint detection')
%Form the squared euclidean distance matrix
DC=zeros(size(S1,2),size(S2,2));
%Form the cumulative distance matrix
DC(1,1)=(S1(1)-S2(1))^2;
DC(1,2)=(S1(1)-S2(1))^2+(S1(1)-S2(2))^2;
DC(2,1)=(S1(1)-S2(1))^2+(S1(2)-S2(1))^2;
for i=3:1:length(S1)
    DC(i,1)=DC(i-1,1)+(S1(i)-S2(1))^2;
end
for i=3:1:length(S2)
    DC(1,i)=DC(1,i-1)+(S1(1)-S2(i))^2;
end
for i=2:1:length(S1)
    for j=2:1:length(S2)
        temp=[DC(i-1,j) DC(i,j-1) DC(i-1,j-1)];
        DC(i,j)=min(temp)+(S1(i)-S2(j))^2;
    end
end
%Argument collection
% 3.2 Dynamic Time Warping 99
n=2;
POS{1}=[length(S1) length(S2)];
ITER=1;
i=length(S1);
j=length(S2);
while(ITER==1)
    if((i==1)&&(j~=1))
        POS{n}=[1 j-1];
        i=POS{n}(1);
        j=POS{n}(2);
        n=n+1;
        break
    elseif((j==1)&&(i~=1))
        POS{n}=[i-1 1];
        i=POS{n}(1);
        j=POS{n}(2);
        n=n+1;
    elseif((j==1)&&(i==1))
        POS{n}=[1 1];
        ITER=2;
        n=n+1;
        break
    else
        [p,q]=min([DC(i-1,j-1) DC(i-1,j) DC(i,j-1)]);
        if(q==1)
            POS{n}=[i-1 j-1];
        elseif(q==2)
            POS{n}=[i-1 j];
        elseif(q==3)
            POS{n}=[i j-1];
        end
        i=POS{n}(1);
        j=POS{n}(2);
        n=n+1;
    end
end
%Speech warping
POS1=cell2mat(POS');
spos1=size(POS1,1);
POS1=[POS1;ones(1,POS1(spos1,2))' (POS1(spos1,2):-1:1)'];
for i=1:1:length(S2)
    [p,q]=find(POS1(:,2)==i);
    [r,s]=min(abs(POS1(p,2)-i));
    temp(i)=S1(1,POS1(p(s)))
    %     100 3 Feature Extraction of the Speech Signal
end
figure
subplot(3,1,1)
plot(S2)
title('Reference speech signal')
subplot(3,1,2)
plot(S1,'r')
title('Query speech signal')
subplot(3,1,3)
plot(temp,'g')
title('Query speech signal after dynamic time warping')
end