d=5; %distance
h=0.1; %head radius
Fs=44100;
Ft=600;
note=tonegen(Ft, Fs, 0.10);
note=note+tonegen(Ft*2, Fs, 0.1);
%Speed of sound
Vs=350; %m/s
ln=length(note);
%Cosine rule constants
b2c2=d^2 + h^2;
b2c=2*d*h;
for theta=-pi:pi/20:pi
    %Calculate path differences
    lp= b2c2+b2c*cos((pi/2)+theta);
    rp= b2c2+b2c*cos((pi/2)-theta);
    %Calculate sound travel times
    lt= lp/Vs;
    rt= rp/Vs;

    %How many samples is this at sample rate Fs
    ls= round(Fs*lt);
    rs= round(Fs*rt);
    %Handle each side separately
    if(rs>ls) %right is further
        df=rs-ls;
        left=[note, zeros(1,df)]/ls;
        right=[zeros(1,df),note]/rs;
    else %left is further
        df=ls-rs;
        left=[zeros(1,df),note]/ls;
        right=[note, zeros(1,df)]/rs;
    end
    %Create the output matrix
    audio=[left;right];
    soundsc(audio, Fs);
    pause(0.2);
end

%%

Lx=1.6;% horiz dimensions, in m.
Ly=1;% vert dimensions, in m.
Xp=1; Yp=0.5;%Xp, Yp: coordinates of the sound source
W=0.2; % about 1.7 kHz %W: wavelength of the sound (where W=340/freq)
pix=1/1000; % 1 mm ->the size of one point to simulate, in m.
arrR=pt_srce_sim(Lx,Ly,pix,0.45,0,W);
arrL=pt_srce_sim(Lx,Ly,pix,1.15,0,W);
%display image
imagesc(arrR+arrL)
colormap('gray') %using greyscale colours
%save to file
imwrite(arrR+arrL,'stereo_source.tiff');