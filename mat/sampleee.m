clear all
clc
close all
% The following MATLAB code creates some simple two-dimensional test data for
% a binary classification demonstration.
% Straight line classifier and the perceptron
%Make a 6x6 square matrix of 2D training data
xin=[[reshape(repmat([0:0.2:1],6,1),[1,6*6])];[repmat([0:0.2:1],1,6)]]';
N = length(xin);
%most are class 0, except a few at lower left
yout=zeros(length(xin));
yout(1:12)=1;
yout(13:15)=1;

%Define a 2D perceptron
b = -0.5;
nu = 0.5;
w = 0.5*rand(3,1); %-1*2.*rand(3,1);
iterations = 100;

%Loop around the training loop
for i = 1:iterations
    for j = 1:N
        y = b*w(1,1)+xin(j,1)*w(2,1)+xin(j,2)*w(3,1);
        z(j) = 1/(1+exp(-2*y));
        delta = yout(j)-z(j);
        w(1,1) = w(1,1)+nu*b*delta;
        w(2,1) = w(2,1)+nu*xin(j,1)*delta;
        w(3,1) = w(3,1)+nu*xin(j,2)*delta;
    end
end
% The code above works well enough (given sufficient iterations) to train a perceptron to
% model any linearly separable classification on the input data:
% It is much easier to understand this if we can visualise the process. To do so, we can
% create two subplots, one of which plots the correct classification of the training data:
figure
subplot(2,1,1)
hold on
for i=1:N
    if(yout(i)==1)
        plot(xin(i,1),xin(i,2),'ko') %class 1
    else
        plot(xin(i,1),xin(i,2),'ro') %class 0
    end
end


NX=1024; %x resolution
NY=1024; %y resolution
dsc=zeros(NX,NY); % plot surface
%classify over this surface
for x=1:NX
    dx=(x-1)/(NX-1); %range 0:1
    for y=1:NY
        dy=(y-1)/(NY-1); %range 0:1
        z=b*w(1,1)+dx*w(2,1)+dy*w(3,1);
        dsc(x,y)=1/(1+exp(-2*z));
    end
end

%% classify the original xin points
for j = 1:N
    y=b*w(1,1)+xin(j,1)*w(2,1)+xin(j,2)*w(3,1);
    z(j) = 1/(1+exp(-2*y));
end
%plot the xin points and classifications
hold on
subplot(2,1,2)
for i=1:N
    hold on
    if(z(i)>0.5)
        plot(xin(i,1)*NX,xin(i,2)*NY,'k*') %class 1
    else
        plot(xin(i,1)*NX,xin(i,2)*NY,'r*') %class 0
    end
end
%overlay the decision line on the plot
contour(transpose(dsc > 0.5),1)


%%

[sp,fs]=audioread('me_and_nicole.WAV'); %load speech
sp=sp./max(abs(sp)); %normalise
sp = sp(:,1);
N=2;
T=length(sp);
tone1=tonegen(440,fs,T/fs); %create a tone
M=2; %create 2 mixtures of these sounds
mix1=sp'*0.23+tone1*0.78;
mix2=sp'*0.81+tone1*0.09;
X=[mix1 ; mix2]; %mixture matrix
% soundsc(mix1,fs)
% soundsc(mix2,fs)
%dimension of X is [M][T]

% Next we create an ICA loop beginning with randomly initialisedWmatrix, then carry
% out step-by-step refining this until either the maximum number of iterations has been
% reached or the convergence criterion has been met:
nu=0.0001; %learning rate
conv=1e-09; %convergence criterion
%---create an [N][M] sized matrix W
W=rand(N,M)*0.1;
%---estimate the source, Sp, dim [N][t]
iter=1;
max_iter=500000;
while(iter < max_iter)
    Sp=W*X;
    %     Find the overall distance, Zn,t = g(yn,t) for n = 1, . . . ,N and t = 1, . . . , T.
    Z=zeros(N,T);
    for n=1:N
        Z(n,:)=1./(1+exp(-Sp(n,:)/2));
    end
    %----calculate DW (deltaW)
    DW=nu*(eye(N) + (1-2*Z)*transpose(Sp))*W;
    %---terminate if converged
    delta=abs(sum(sum(DW)));
    if(delta < conv)
        break;
    end
    %print out (so we know what's happening)
    if mod(iter,10000)==0
        fprintf('Iteration %d, sum=%g\n',iter,delta);
    end
    %---apply the update
    W=W+DW;
    iter=iter+1;
end

% check the seperation:
% soundsc(Sp(1,:),fs)
% soundsc(Sp(2,:),fs)



%%
M=40;
figure
% Create some 2D data in 3 classes
c1=([1.9*ones(M,1) 0.9*ones(M,1)])+randn(M,2);
c2=([0.9*ones(M,1) 0.1*ones(M,1)])+randn(M,2);
c3=([0.1*ones(M,1) 1.9*ones(M,1)])+randn(M,2);

% Plot the data
plot(c1(:,1),c1(:,2),'r+')
hold on
plot(c2(:,1),c2(:,2),'bx')
plot(c3(:,1),c3(:,2),'g*')
X=[c1;c2;c3];
% C=kmeans(X',3);
[C,~]=GMMlearn(X,3)
plot(C(1,1),C(1,2),'ko')
plot(C(2,1),C(2,2),'ko')
plot(C(3,1),C(3,2),'ko')
hold off

%%
clear all
clc

% Code to perform this VAD(voice activity detection) in MATLAB is as follows:
%the noisy speech is in array nspeech
%fs is the sample rate
[nspeech,fs]=audioread('me_and_nicole.WAV'); %load speech
nspeech = nspeech(:,1);
nspeech = nspeech(round(1.4*fs):round(4*fs));
alpha=0.98;
WindowLength=10*10^-3;  % 30 [mS] window
Overlap=50;             % 50% overlap
[ProcessedSig,FramedSig] = PreProcess(nspeech,fs,alpha,WindowLength,Overlap);
mse=mean((nspeech-ProcessedSig).^2)
L=length(ProcessedSig);
frame= 10*10^-3; %0.1 frame size in seconds
frame=floor(fs*frame); %length
Nf=floor(L/frame); %no. of frames
energy=[];
%plot the noisy speech waveform
figure
subplot(2,1,1)
plot([0:L-1]/fs,ProcessedSig,'DisplayName','speech');axis tight
xlabel('Time,s'); ylabel('Amplitude');legend
%divide into frames, get energy
for n=1:Nf
    seg=ProcessedSig(1+(n-1)*frame:n*frame);
    energy(n)=sum(seg.^2);
end
%plot the energy
subplot(2,1,2)
plot(([1:Nf]*frame)/fs,energy,'g','DisplayName','ENG');
A=axis; A(2)=(Nf-1)*frame/fs; axis(A)
xlabel('Time,s'); ylabel('Energy');
%find the maximum energy, and threshold
emax=max(energy); emin=min(energy);
e05=emin+0.05*(emax-emin);
%draw the threshold on the graph
line(([0 Nf-1]*frame)/fs,[e05 e05],'DisplayName','e05')
A=axis; A(2)=(Nf-1)*frame/fs; axis(A)
%plot the decision (frames > 10%)
hold on
plot([1:Nf]*frame/fs,(energy>e05)*(emax),'r','DisplayName','decision')
A=axis; A(2)=(Nf-1)*frame/fs; axis(A)
hold off; legend

%%

frame= 10*10^-3; %0.1 frame size in seconds
frame=floor(fs*frame); %length
Ws=480;
L=length(ProcessedSig);
Nf=floor(L/Ws);
Bs=[]; %collect pitch multipliers
Ms=[]; %collect pitch taps
% 6.3 Pitch models 181
for ff=1:Nf
    seg=ProcessedSig(1+Ws*(ff-1):Ws*ff).*hamming(Ws);
    [B,M]=ltp(seg);
    Bs=[Bs, B];
    Ms=[Ms, M];
end

figure
subplot(3,1,1)
plot([0:L-1]/fs,ProcessedSig,'DisplayName','speech');axis tight
xlabel('Time,s'); ylabel('Amplitude');legend
subplot(3,1,2)
plot(([1:Nf]*Ws)/fs,Bs,'DisplayName','B multiplier');
A=axis; A(2)=(Nf-1)*frame/fs; axis(A)
xlabel('Time,s'); ylabel('B multiplier');legend
subplot(3,1,3)
plot(([1:Nf]*Ws)/fs,Ms,'DisplayName','M tap');
A=axis; A(2)=(Nf-1)*frame/fs; axis(A)
xlabel('Time,s'); ylabel('M tap');legend