function [Formantss1,LPc_dB,F] =estimatePhonemeFormants(PhonemeSig,Fs,phonemeName,flag)
% % estimatePhonemeFormants takes the PhonemeSig and uses the Power spectral
% % density by calculating the P coefficients using AR LPC model estimation of
% % A general discrete-time model for speech production.
% % Input:
% % PhonemeSig - one phoneme (after pre-processing)
% % Fs - sampling frequency
% % phonemeName - a string with the phoneme name
% % OUTPUT:
% % h1 - handle for spectral estimation graph with Formants values
% % h2 - handle for zero-pole map with chosen poles
% % rng default
% p = Fs/1000 + 2; %model order
% r = xcorr(PhonemeSig); %auto corrolation vector
% [outAR,~] = lpc(PhonemeSig,p); %lpc coefficients
% G = r(1)-sum(outAR(2:end)'.*r(1:p));
%
% [H,w]=freqz(G,outAR); %frequency of LPC model and phonem
% [~,loc] = findpeaks(20*log(abs(H)));
% Formantss1 = w(loc)*Fs/(2*pi);

[Peri,omeg]=periodogram(PhonemeSig,[],'onesided');

P =Fs/1000; %20;
% g-variance of the prediction error
% [aa,gg]=lpc(PhonemeSig,P);
% std_g = sqrt(gg);
% M = length(PhonemeSig);
% % use the Random Noise Generator on samples
% add_noise_to_filt=filtfilt(1,aa,std_g*randn(M,1));
% Autoregressive all-pole model parameters — Yule-Walker method
[aa1,gg1]=aryule(PhonemeSig,P);
std_gg1 = sqrt(gg1);
% add the noise to the parametric periodogram
[hh,omegg1]=freqz(std_gg1,aa1);
LPc_dB = 20*log10(abs(hh));
F = (Fs/(2*pi))*omegg1;
if flag==1

    close all
    figure(100);subplot(2,1,1);
    plot(F,LPc_dB,'LineWidth',1.4,'Color','r',"DisplayName","LPC Spectral Estimation");
    xlabel 'Frequency (Hz)' ;ylabel 'Magnitude [dB]'; hold on; grid on
    plot((Fs/(2*pi))*omeg,10*log10(Peri),'Color','b',"DisplayName","Periodogram")
    xlabel 'Frequency [Hz]' ; ylabel 'Gain [dB]'
    hold on
end
% find formants:
rts = roots(aa1);
%Because the LPC coefficients are real-valued, the roots occur in complex
% conjugate pairs. Retain only the roots with one sign for the imaginary
% part and determine the angles corresponding to the roots.
rts = rts(imag(rts)>0);

angz = atan2(imag(rts),real(rts));
rts = rts(angz<((6*pi)/8)); % limit the serch untill 6[KHz]
angz = angz(angz<((6*pi)/8));

bw1 = -1/2*(Fs/(2*pi))*log(abs(rts));

[max_rts,Poles_idx]=maxk(rts,4,'ComparisonMethod','abs');
Formantss1 = (angz(Poles_idx).*(Fs/(2*pi)))';
Formantss1 = sort(Formantss1);


% [~,Poles_idx]=maxk(rts,2,'ComparisonMethod','abs');
% Formantss1(1)=Formantss(1);
% Formantss1(2:3) = (angz(Poles_idx).*(Fs/(2*pi)))';
% Formantss1 = sort(Formantss1);

% 
% 
% % % Convert the angular frequencies in rad/sample represented by the angles to hertz and calculate the bandwidths of the formants.
% % % The bandwidths of the formants are represented by the distance of the prediction polynomial zeros from the unit circle.
% [frqs,indices] = sort(angz.*(Fs/(2*pi)));
% bw = -1/2*(Fs/(2*pi))*log(abs(rts(indices)));
% 
% Formantss =[ NaN NaN NaN] ;
% % Use the criterion that formant frequencies   && bw(kk) >10 && bw(kk) <400  should be greater than 90 Hz with bandwidths less than 400 Hz to determine the formants.
% nn = 1;
% for kk = 1:length(frqs)
%     if (frqs(kk) > 90 && nn<=3)
%         if frqs(kk)<1050 && (nn==1  )&& bw(kk) >10
%             Formantss(nn) = frqs(kk);
%             nn = nn+1;
%         elseif frqs(kk)>1000 && frqs(kk)<3000 && (nn==2  )&& bw(kk) >10
%             Formantss(nn) = frqs(kk);
%             nn = nn+1;
%         elseif frqs(kk)>3000 && frqs(kk)<6000 && ( nn==3 ) && bw(kk) >10
%             Formantss(nn) = frqs(kk);
%             nn = nn+1;
%         end
%         if nn ==4
%             break
%         end
%     end
% end
% if size(Formantss,2)~=3
%     Formantss = Formantss1 ;
% end

% Formantss
%
% Formantss1 = angz(Poles_idx).*(Fs/(2*pi));%
% Formantss1 = sort(Formantss1)';
% H = tf(1,aa1,1/Fs);
% [p,~] = pzmap(H);
% pos_imag = p(imag(p)>0);
% pos_real = pos_imag(real(pos_imag)>0);
% pos_real = sort(pos_real,'descend','ComparisonMethod','real');
% Fpoles = pos_real(1:3);
% angles = atan2(imag(Fpoles),real(Fpoles));
% formants2 = (angles.*(Fs/(2*pi)))';
if flag==1
    figure(100);
    hold on
    %     title(sprintf('Spectral Estimation of the Phenome %s',phonemeName))
    title(sprintf('Estimated Formants F1=%d Hz, F2=%d Hz, F3=%d Hz',round(Formantss1(1)),...
        round(Formantss1(2)),round( Formantss1(3))))
    legend()
    %     annotation(h1,'textbox',[ 0.15 0.09 0.09 0.1],'String',...
    %         {sprintf('Estimated Formants F1=%d Hz, F2=%d Hz, F3=%d Hz',round(Formantss(1)),...
    %         round(Formantss(2)),round( Formantss(3)))},'FitBoxToText','on');grid on
end

if flag==1

    figure(100);subplot(2,1,2)
    [z,pp,k]  = zplane(1,aa1); hold on; grid on
    findzeros = findobj(z,'Type','line'); findpoles = findobj(pp,'Type','line');
    findk = findobj(k,'Type','line');
    set(findzeros,'Color','b','linewidth',1.1,'markersize',9);
    set(findpoles,'Color','b','linewidth',1.1,'markersize',9);
    set(findk,'Color','b','linewidth',1.1,'markersize',9);
    hold on
    [z1,~,~]=zplane(rts(Poles_idx(1:3)));
    findzeros1 = findobj(z1,'Type','line');
    set(findzeros1,'Color','r','linewidth',1.2,'markersize',9);
    xlim([-1.1 1.1]);ylim([-1.1 1.1]);grid on
    title(sprintf('AR Model Poles- using LPC- Phoneme %s',phonemeName))
    legend('LPC Zeros','LPC Poles','Z-plane Axes','Chosen Poles','Location','southeast')
end

end
%
%
