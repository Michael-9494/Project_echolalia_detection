function [h1,h2]=estimatePhonemeFormants(PhonemeSig,Fs,phonemeName)
% estimatePhonemeFormants takes the PhonemeSig and uses the Power spectral 
% density by calculating the P coefficients using AR LPC model estimation of 
% A general discrete-time model for speech production. 
% Input:
% PhonemeSig - one phoneme (after pre-processing)
% Fs - sampling frequency
% phonemeName - a string with the phoneme name
% OUTPUT:
% h1 - handle for spectral estimation graph with Formants values
% h2 - handle for zero-pole map with chosen poles
rng default

[Peri,omeg]=periodogram(PhonemeSig,[],'onesided');

P = Fs/1000 +2;
% g-variance of the prediction error
[aa,gg]=lpc(PhonemeSig,P);
std_g = sqrt(gg);
M = length(PhonemeSig);
%use the Random Noise Generator on samples
add_noise_to_filt=filter(1,aa,std_g*randn(M,1));
% Autoregressive all-pole model parameters — Yule-Walker method
[aa1,gg1]=aryule(add_noise_to_filt,P);
std_gg1 = sqrt(gg1);
% add the noise to the parametric periodogram
[hh,omegg1]=freqz(std_gg1,aa1);

h1 = figure;
plot((Fs/(2*pi))*omegg1,20*log10(abs(hh)),'LineWidth',1.4,'Color','r',"DisplayName","LPC Spectral Estimation");
xlabel 'Frequency (Hz)' ;ylabel 'Magnitude [dB]'; hold on; grid on
plot((Fs/(2*pi))*omeg,10*log10(Peri),'Color','b',"DisplayName","Periodogram")
xlabel 'Frequency [Hz]' ; ylabel 'Gain [dB]'
hold on

% find formants:
rts = roots(aa1);
%Because the LPC coefficients are real-valued, the roots occur in complex
% conjugate pairs. Retain only the roots with one sign for the imaginary
% part and determine the angles corresponding to the roots.
rts = rts(imag(rts)>=0);
[~,Poles_idx]=maxk(abs(rts),3);
angz = atan2(imag(rts),real(rts));
% % Convert the angular frequencies in rad/sample represented by the angles to hertz and calculate the bandwidths of the formants.
% % The bandwidths of the formants are represented by the distance of the prediction polynomial zeros from the unit circle.
% [frqs,indices] = sort(angz.*(Fs/(2*pi)));
% bw = -1/2*(Fs/(2*pi))*log(abs(rts(indices)));
% 
% % Use the criterion that formant frequencies should be greater than 90 Hz with bandwidths less than 400 Hz to determine the formants.
% nn = 1;
% for kk = 1:length(frqs)
%     if (frqs(kk) > 90 && bw(kk) <400)
%         formants(nn) = frqs(kk);
%         nn = nn+1;
%     end
% end
% formants

Formantss = angz(Poles_idx).*(Fs/(2*pi));
Formantss = sort(Formantss);
hold on
title(sprintf('Spectral Estimation of the Phenome %s',phonemeName))
legend()
annotation(h1,'textbox',[ 0.15 0.09 0.09 0.1],'String',...
 {sprintf('Estimated Formants F1=%d Hz, F2=%d Hz, F3=%d Hz',round(Formantss(1)),...
 round(Formantss(2)),round( Formantss(3)))},'FitBoxToText','on');grid on


h2 = figure;
[z,p,k]  = zplane(1,aa1); hold on; grid on
findzeros = findobj(z,'Type','line'); findpoles = findobj(p,'Type','line');
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

% A = lpc(x1,8);
% rts = roots(A);
% 
% % Because the LPC coefficients are real-valued, the roots occur in complex conjugate pairs. Retain only the roots with one sign for the imaginary part and determine the angles corresponding to the roots.
% rts = rts(imag(rts)>=0);
% angz = atan2(imag(rts),real(rts));
% % 
