close all 
clc
clear all

% Example of E-M algorithm
% Three clusters of Gaussian data
N=100;

%cluster 1
mux1=3; muy1=3; %mean
sigmax1=1; sigmay1=0.4; %Standard deviation
% Normal random numbers
x1=normrnd(mux1,sigmax1,1,N);
y1=normrnd(muy1,sigmay1,1,N);
%cluster 2
mux2=10; muy2=4; sigmax2=1; sigmay2=0.8;
x2=normrnd(mux2,sigmax2,1,N);
y2=normrnd(muy2,sigmay2,1,N);
%cluster 3
mux3=6; muy3=7; sigmax3=1; sigmay3=0.6;
x3=normrnd(mux3,sigmax3,1,N);
y3=normrnd(muy3,sigmay3,1,N);

% complete data set
D=zeros(2,3*N);
D(1,:)=[x1,x2,x3];
D(2,:)=[y1,y2,y3];

% centroids: initial guess (3 clusters)
mu=zeros(2,3);
mu(1,1)=D(1,1);mu(2,1)=D(2,1);
mu(1,2)=D(1,120); mu(2,2)=D(2,120);
mu(1,3)=D(1,215); mu(2,3)=D(2,215);
%variance: initial guess
si=2*ones(2,3);
pd=zeros(3,3*N); %for PDF value comparison
mL=zeros(3*N,1); %membership label
%iterations------------------------------------------------
for it=1:100
    mn=zeros(2,3); %new centroid
    nk=zeros(1,3);
    %step E
    %values of PDFs at each datum (vectorized code)
    pd(1,:)=(normpdf(D(1,:),mu(1,1),si(1,1))).*normpdf(D(2,:),...
        mu(2,1),si(2,1));
    pd(2,:)=(normpdf(D(1,:),mu(1,2),si(1,2))).*normpdf(D(2,:),...
        mu(2,2),si(2,2));
    pd(3,:)=(normpdf(D(1,:),mu(1,3),si(1,3))).*normpdf(D(2,:),...
        mu(2,3),si(2,3));
    for nn=1:(3*N)
        k=1; v1=pd(1,nn); v2= pd(2,nn); v3= pd(3,nn);
        if v1<v2, k=2; end
        if(k==1 && v1<v3), k=3; end
        if(k==2 && v2<v3), k=3; end
        mL(nn)=k; %assign membership label;
        mn(:,k)=mn(:,k)+D(:,nn); nk(k)=nk(k)+1; %accumulate
    end
    %step M
    %new centroids
    mn(1,:)=mn(1,:)./nk(1,:); %average
    mn(2,:)=mn(2,:)./nk(1,:); % " "
    %new variances
    for nn=1:(3*N)
        k=mL(nn); %read label
        si(1,k)=si(1,k)+((D(1,nn)-mn(1,k))^2);
        si(2,k)=si(2,k)+((D(2,nn)-mn(2,k))^2);
    end
    for n=1:3
        si(1,n)=sqrt(si(1,n)/nk(n)); si(2,n)=sqrt(si(2,n)/nk(n));
    end
    mu=mn; %change of centroid
end
%--------------------------------------------------------------
%prepare contour display
p=0:0.2:100;
px1=normpdf(p,mu(1,1),si(1,1));py1=normpdf(p,mu(2,1),si(2,1));
pz1=px1'*py1; %matrix
px2=normpdf(p,mu(1,2),si(1,2));py2=normpdf(p,mu(2,2),si(2,2));
pz2=px2'*py2; %matrix
px3=normpdf(p,mu(1,3),si(1,3));py3=normpdf(p,mu(2,3),si(2,3));
pz3=px3'*py3; %matrix
%display
figure(1)
scatter(D(1,:),D(2,:),32,'ro'); hold on; %the data
axis([0 14 0 9]);
plot(mu(1,:),mu(2,:),'b*','MarkerSize',16); hold on; %centroids
contour(p,p,pz1',6); %gaussian PDF
contour(p,p,pz2',6); %" " "
contour(p,p,pz3',6); %" " "
title('Classification with EM')
xlabel('x'); ylabel('y');
%print mu and sigma for each cluster
mu
si

%%
clear all
clc

% Bayesian regression example
% generate a data set
m=0.4; %slope
b=0.6; %intercept
N=25; %number of data points
x=10*rand(N,1);
std=0.2;
% 760 7 Data Analysis and Classification
nse=normrnd(0,std,N,1); %noise
y=m*x+b+nse;
%add second column of 1's for the intercept:
z=cat(2,x,ones(N,1));
% PDF of line parameters
D=zeros(2,1);
gamma=[0.2 0;0 0.6]; %(edit the diagonal numbers)
aux1=(z'*z)/(std^2); aux2=inv(gamma^2);
D=inv(aux1+aux2);
rpar=(D*z'*y)/(std^2);
rmu=rpar(1); rb=rpar(2);
rstd=D;
% Points of the PDF of line parameters
x1=0:0.02:2;
x2=0:0.02:2;
L=length(x1);
dd=det(rstd);
K=1/(2*pi*sqrt(dd)); Q=1/2;
ypdf=zeros(L,L); %space for the PDF
for ni=1:L
    for nj=1:L
        aux=(((x1(ni)-rmu)^2)/rstd(1,1))+(((x2(nj)-rb)^2)/rstd(2,2));
        ypdf(ni,nj)= K*exp(-Q*aux);
    end
end
% display ------------------
figure(1)
contour(x1,x2,ypdf);
axis([0 1 0 0.8]);
grid;
title('PDF of line parameters');
xlabel('intercept'); ylabel('slope');
figure(2)
plot(x,y,'r*'); hold on;
bx0=0; by0=rb;
bxf=10; byf=rmu*bxf+rb;
plot([bx0 bxf],[by0 byf],'k');
title('Bayesian regression');
xlabel('x'); ylabel('y');

%%
clear all
clc
% Bayesian prediction/interpolation example
% generate a data set
m=0.4; %slope
b=0.6; %intercept
N=25; %number of data points
x=10*rand(N,1);
std=0.2;
% 7.6 Classification and Probabilities 763
nse=normrnd(0,std,N,1); %noise
y=m*x+b+nse;
%add second column of 1's for the intercept:
z=cat(2,x,ones(N,1));
% PDF of line parameters
D=zeros(2,1);
gamma=[0.2 0;0 0.6]; %(edit the diagonal numbers)
aux1=(z'*z)/(std^2); aux2=inv(gamma^2);
D=inv(aux1+aux2);
rpar=(D*z'*y)/(std^2);
rmu=rpar(1); rb=rpar(2);
rstd=D;
% new datum
nz=[5;1];
% PDF of predicted ny
ny=(nz'*D*z'*y)/(std^2);
rstd=(nz'*D*nz)+(std^2);
% Points of the PDF of predicted ny
x1=0:0.01:4;
L=length(x1);
K=1/(rstd*sqrt(2*pi)); Q=0.5;
ypdf=zeros(L,1); %space for the PDF
for ni=1:L
    aux=((x1(ni)-ny)^2)/(rstd^2);
    ypdf(ni,1)= K*exp(-Q*aux);
end
% display ------------------
figure(1)
plot(x,y,'r*'); hold on;
bx0=0; by0=rb;
bxf=10; byf=rmu*bxf+rb;
plot([bx0 bxf],[by0 byf],'k-');
plot(nz(1),ny,'bd','MarkerSize',10); %the predicted point
title('Bayesian prediction');
xlabel('x'); ylabel('y');
figure(2)
for ni=10:L-1
    plot([x1(ni) x1(ni+1)],[ypdf(ni) ypdf(ni+1)],'b'); hold on;
end
title('PDF of the predicted point');
xlabel('predicted y value')

%%
clear all
clc
% Gauss Process samples
% Build a kernel matrix
sigmaf=1.2;
gammaf=0.9; qf=2*(gammaf^2);
sigman=0.1;
x=-2:0.02:2; L=length(x);
K=zeros(L,L);
for i=1:L
    for j=i:L
        nse=(sigman^2)*(x(i)==x(j)); %noise term
        K(i,j)=((sigmaf^2)*exp(-((x(i)-x(j))^2)/qf))+nse;
        K(j,i)=K(i,j);
    end
    % 774 7 Data Analysis and Classification
end
% prepare for sampling
[V,D]= eig(K);
A=V*sqrt(D);
% take 9 samples
rv=zeros(L,9);
for nn=1:9
    rv(:,nn)=A*randn(L,1);
end
figure(1)
subplot(1,2,1)
imagesc(K);
title('Kernel matrix');
subplot(1,2,2)
plot(x,rv);
title('Nine samples of the Gaussian Process');
xlabel('x'); ylabel('y');

%%
clear all
clc
% Scatterplot of original sources and
% scatterplot of mixed signals
% example of two speeches
%read two sound files
[a,fs1]=audioread('spch1.wav'); %read wav file
[b,fs1]=audioread('spch2.wav'); % " " "
a = a(1:10000);
b = b(1:length(a));
R=2; %reduce data size for clearer picture
a=decimate(a,R);
b=decimate(b,R);
s1=(a-mean(a))'; %zero-mean
s2=(b-mean(b))'; % " " "
vr1=var(s1); s1=s1/sqrt(vr1); %variance=1
vr2=var(s2); s2=s2/sqrt(vr2); %" " "
% 7.4 Independent Component Analysis (ICA) 667
s=[s1';s2']; %combine sources
%mix of sources
N=length(s1);
M=[0.7 0.3; 0.2 0.8]; %example of mixing matrix
x=M*s; %mixed signals
%display
figure(1)
subplot(1,2,1);
%scatterplot of sources
plot(s(1,:),s(2,:),'k.'); hold on; %scatterplot
L=3;
%axes
plot([-L L],[0 0],'k');
plot([0 0],[-L L],'k');
axis([-L L -L L]);
title('scatterplot of 2 sources');
xlabel('s1'); ylabel('s2');
subplot(1,2,2);
%scatterplot of mixed signals
plot(x(1,:),x(2,:),'k.'); hold on; %scatterplot
L=3;
%axes
plot([-L L],[0 0],'k');
plot([0 0],[-L L],'k');
axis([-L L -L L]);
title('scatterplot of mixed signals');
xlabel('s1'); ylabel('s2');

%%
clear all
clc
% Comparison of PCA and ICA components
% example of two mixed speeches
%read two sound files
[a,fs1]=audioread('spch1.wav'); %read wav file
[b,fs1]=audioread('spch2.wav'); % " " "
b = b(1:512);
a = a(1:length(b));
R=2; %reduce data size for clearer picture
a=decimate(a,R);
b=decimate(b,R);
s1=(a-mean(a))'; %zero-mean
s2=(b-mean(b))'; % " " "
vr1=var(s1); s1=s1/sqrt(vr1); %variance=1
vr2=var(s2); s2=s2/sqrt(vr2); %" " "
s=[s1';s2']; %combine sources
% 670 7 Data Analysis and Classification
%mix of sources
N=length(s1);
M=[0.7 0.3; 0.2 0.8]; %example of mixing matrix
x=M*s; %mixed signals
% PCA computation
A=x'/sqrt(N-1);
%singular value decomposition
[U,S,V]=svd(A); %V contains principal components
%ICA computation
U=inv(M);
w1=U(1,:);
w2=U(2,:);
%display
figure(1)
%scatterplot and PCA components
plot(x(1,:),x(2,:),'k.'); hold on; %scatterplot
L=3;
%PCA components:
plot([-L*V(1,1) L*V(1,1)],[-L*V(2,1) L*V(2,1)],'r');
plot([-L*V(1,2) L*V(1,2)],[-L*V(2,2) L*V(2,2)],'r');
%axes
plot([-L L],[0 0],'k');
plot([0 0],[-L L],'k');
axis([-L L -L L]);
title('scatterplot of 2 mixed speeches: PCA components');
xlabel('x'); ylabel('y');
figure(2)
%scatterplot and ICA components (perpendicular to w)
plot(x(1,:),x(2,:),'k.'); hold on; %scatterplot
L=3;
%ICA components:
plot([-L*w1(2) L*w1(2)],[L*w1(1) -L*w1(1)],'r');
plot([-L*w2(2) L*w2(2)],[L*w2(1) -L*w2(1)],'r');
%axes
plot([-L L],[0 0],'k');
plot([0 0],[-L L],'k');
axis([-L L -L L]);
title('scatterplot of 2 mixed speeches: ICA components');
xlabel('x'); ylabel('y');

%%
% Kurtosis projection pursuit
% example of two original speeches
%read two sound files
[a,fs1]=audioread('spch1.wav'); %read wav file
[b,fs1]=audioread('spch2.wav'); % " " "
b = b(1:512);
a = a(1:length(b));
s1=(a-mean(a))'; %zero-mean
s2=(b-mean(b))'; % " " "
vr1=var(s1); s1=s1/sqrt(vr1); %variance=1
vr2=var(s2); s2=s2/sqrt(vr2); %" " "
s=[s1';s2']; %combine sources
N=length(a);
A=60; %number of circle partitions
kur=zeros(1,A+1);
% 684 7 Data Analysis and Classification
ag=zeros(1,A+1);
i=1:N; %vectorized iterations
for nn=0:A
    alpha=(2*pi*nn)/A; %angle of projection axis in radians
    p=zeros(1,N);
    %projection of scatterplot on the inclined axis
    p(i)=(s(1,i)*cos(alpha)) + (s(2,i)*sin(alpha));
    moment4=mean(p.^4); %fourth moment
    moment2=mean(p.^2); %second moment
    kst=moment4-(3*(moment2.^2)); %kurtosis
    kur(nn+1)=kst; %save result
    ag(nn+1)=alpha;
end
%display
%pursuit
figure(2)
polar(ag,kur,'k');
title('kurtosis as projection axis rotates');


%%

% Kurtosis projection pursuit
% example of two mixed speeches
%read two sound files


clear all
clc
% Comparison of PCA and ICA components
% example of two mixed speeches
%read two sound files
[a,fs1]=audioread('spch1.wav'); %read wav file
[b,fs1]=audioread('spch2.wav'); % " " "
b = b(1:512);
a = a(1:length(b));
s1=(a-mean(a))'; %zero-mean
s2=(b-mean(b))'; % " " "
vr1=var(s1); s1=s1/sqrt(vr1); %variance=1
% 686 7 Data Analysis and Classification
vr2=var(s2); s2=s2/sqrt(vr2); %" " "
s=[s1';s2']; %combine sources
%mix of sources
N=length(s1);
M=[0.7 0.3; 0.3 0.7]; %example of mixing matrix
x=M*s; %mixed signals
N=length(a);
A=60; %number of circle partitions
kur=zeros(1,A+1);
ag=zeros(1,A+1);
i=1:N; %vectorized iterations
for nn=0:A
    alpha=(2*pi*nn)/A; %angle of projection axis in radians
    p=zeros(1,N);
    %projection of scatterplot on the inclined axis
    p(i)=(x(1,i)*cos(alpha)) + (x(2,i)*sin(alpha));
    moment4=mean(p.^4); %fourth moment
    moment2=mean(p.^2); %second moment
    kst=moment4-(3*(moment2.^2)); %kurtosis
    kur(nn+1)=kst; %save result
    ag(nn+1)=alpha;
end
%display
%scatterplot
figure(1)
plot(x(1,:),x(2,:),'k.'); hold on; %scatterplot
L=2.7;
plot([-L L],[0 0],'k');
plot([0 0],[-L L],'k');
axis([-L L -L L]);
title('scatterplot: 2 mixed speeches');
xlabel('x'); ylabel('y');
%pursuit
figure(2)
polar(ag,kur,'k');
title('kurtosis as projection axis rotates');