function arr=pt_srce_sim(Lx,Ly,pix,Xp,Yp,W)
%Lx, Ly: horiz. and vert. dimensions, in m.
%pix: the size of one point to simulate, in m.
%Xp, Yp: coordinates of the sound source
%W: wavelength of the sound (where W=340/freq)
Nx=floor(Lx/pix);
Ny=floor(Ly/pix);
arr=zeros(Nx,Ny); %define the area
for x=1:Nx
    for y=1:Ny
        px=x*pix;
        py=y*pix;
        d=sqrt((px-Xp).^2+(py-Yp).^2);
        arr(x,y)=cos(2*pi*d/W);
    end
end