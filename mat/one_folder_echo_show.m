clear;close all;clc
Fs = 16000;
flag =1;

load("15_12_2022_25092022.mat")


Param = struct();
% start with feature extraction
Param.alpha=15/16; % for pre emphasis
Param.WindowLength=20*10^-3;  % 30 [mS] window
Param.WindowLenSamp=(Param.WindowLength*Fs);
Param.Overlap=50;             % 50% overlap
% Param.fftLength = 2^nextpow2(Param.WindowLenSamp);
Param.noverlap=(((Param.Overlap)*Param.WindowLength)/100 *Fs);

% audiowrite("child.mp4",all(1).frames_out_child(112).Signal_frame,Fs,'BitRate',256)
F1_ther = [];F2_ther = [];F3_ther = [];
F1_ch = [];F2_ch = [];F3_ch = [];
F1_chWa = [];F2_chWa = [];F3_chWa = [];
alpha = 1.12; % for frequency warping

F_for_spect = linspace(0,Fs/2,161);% generates n points. The spacing between the points is (x2-x1)/(n-1).);
% Recs_for_cry_scream_18092022_ECHO
for file = 1:length(all)
    VTLN =0;
    Eb_T=all(file).NRG_median_therapist;ZC_T = all(file).ZCR_median_therapist;
    Eb_C= all(file).NRG_median_Child; ZC_C = all(file).ZCR_median_Child;
    frames_out_Therapist = Process_frame(all(file).frames_Therapist,Fs,Param,0,VTLN,alpha,Eb_T,Eb_C,ZC_T,ZC_C);
    VTLN = 1;
    frames_out_child = Process_frame(all(file).frames_Child,Fs,Param,0,VTLN,alpha,Eb_T,Eb_C,ZC_T,ZC_C);
    %     F1_ther = [];F2_ther = [];F3_ther = [];%F4_ther = [];
    %     F1_ch = [];F2_ch = [];F3_ch = [];%F4_ch = [];
    %     F1_chWa = [];F2_chWa = [];F3_chWa = [];
    %     for w = 1:length(frames_out_Therapist)
    %         F1_ther = [F1_ther frames_out_Therapist(w).F1];
    %         F2_ther = [F2_ther frames_out_Therapist(w).F2];
    %         F3_ther = [F3_ther frames_out_Therapist(w).F3];
    %         %         F4_ther = [F4_ther frames_out_Therapist(w).F4];
    %
    %     end
    %
    %     for w = 1:length(frames_out_child)
    %         F1_ch = [F1_ch frames_out_child(w).F1];
    %         F2_ch = [F2_ch frames_out_child(w).F2];
    %         F3_ch = [F3_ch frames_out_child(w).F3];
    %         F1_chWa = [F1_chWa frames_out_child(w).F1Warped ];
    %         F2_chWa = [F2_chWa frames_out_child(w).F2Warped ];
    %         F3_chWa = [F3_chWa frames_out_child(w).F3Warped ];
    %         %         F4_ch = [F4_ch frames_out_child(w).F4];
    %     end
    %
    for w = 1:length(frames_out_Therapist)
        if isempty(frames_out_Therapist(w).t)
            continue
        else
            
            
            if ~strcmp( frames_out_Therapist(w).segment_event ,'Echolalia')
                continue
            else
                
                
                % find the closest echo segment of child
                for j = 1:length(frames_out_child)
                    if isempty(frames_out_child(j).t)
                        continue
                    else
                        if (frames_out_Therapist(w).t(end)+15) >(frames_out_child(j).t(1)) &&...
                                ((frames_out_Therapist(w).t(end)) <=(frames_out_child(j).t(1)))&&...
                                strcmp( frames_out_child(j).segment_event{1} ,'Echolalia')
                            echo_idx = j;
                            break
                        end
                    end
                end
                
                
                if flag == 1
%                     pause(5)
                    close all
                    soundsc(all(file).frames_Therapist(w).data,Fs);
                    fh = figure(2);
                    fh.WindowState = 'maximized';
                    
                    figure(2);subplot(3,3,1);
                    plot(frames_out_Therapist(w).t,all(file).frames_Therapist(w).data)
                    title(['speech signal ' frames_out_Therapist(w).segment_event]);
                    xlabel('time[s]');ylabel('amp'); axis tight
                    
                    figure(2);subplot(3,3,4)
                    plot(frames_out_child(echo_idx).t,all(file).frames_Child(echo_idx).data)
                    title(['speech signal ' frames_out_child(echo_idx).segment_event]);
                    xlabel('time[s]');ylabel('amp'); axis tight
                    
                    T = 1/Fs;             % Sampling period
                    Len = length(frames_out_child(echo_idx).warpedSignal);
                    t = frames_out_child(echo_idx).t(1) + (0:Len-1)*T;% Time vector
                    figure(2);subplot(3,3,7); plot(t,frames_out_child(echo_idx).warpedSignal)
                    title(['speech signal warpedSignal ' frames_out_child(echo_idx).segment_event]);
                    xlabel('time[s]');ylabel('amp')
                    axis tight; hold on
                    
                    figure(2);subplot(3,3,2)
                    surf(frames_out_Therapist(w).t_for_spect,F_for_spect,...
                        10*log10(frames_out_Therapist(w).PowerSpectrum),"DisplayName","P_{Spectrum}","EdgeColor","none");
                    view([0,90]); hold on
                    axis([frames_out_Therapist(w).t_for_spect(1) frames_out_Therapist(w).t_for_spect(end) F_for_spect(1) F_for_spect(end)])
                    xlabel('Time (s)');hold on
                    plot(frames_out_Therapist(w).t_for_spect,frames_out_Therapist(w).F1,"b","DisplayName","F_1");
                    hold on;
                    plot(frames_out_Therapist(w).t_for_spect,frames_out_Therapist(w).F2,"r","DisplayName","F_2");
                    hold on;
                    plot(frames_out_Therapist(w).t_for_spect,frames_out_Therapist(w).pitch,"m","DisplayName","F_0");hold on;
                    plot(frames_out_Therapist(w).t_for_spect,frames_out_Therapist(w).F3,"k","DisplayName","F_3");
                    hold on;ylabel('Frequency (Hz)');title("Therapist")
                    
                    
                    figure(2);subplot(3,3,3)
                    surf(frames_out_Therapist(w).t_for_spect,frames_out_Therapist(w).F,...
                        (frames_out_Therapist(w).LPC_mat),"DisplayName","LPC_{Spectrum}","EdgeColor","none");
                    view([0,90]); hold on
                    axis([frames_out_Therapist(w).t_for_spect(1) frames_out_Therapist(w).t_for_spect(end) F_for_spect(1) F_for_spect(end)])
                    xlabel('Time (s)');hold on
                    plot(frames_out_Therapist(w).t_for_spect,frames_out_Therapist(w).F1,"b","DisplayName","F_1");
                    hold on;
                    plot(frames_out_Therapist(w).t_for_spect,frames_out_Therapist(w).F2,"r","DisplayName","F_2");
                    hold on;
                    plot(frames_out_Therapist(w).t_for_spect,frames_out_Therapist(w).F3,"k","DisplayName","F_3");
                    hold on;ylabel('Frequency (Hz)');title("Therapist LPC")
                    
                    pause(2)
                    soundsc(all(file).frames_Child(echo_idx).data,Fs);
                    
                    figure(2);subplot(3,3,5)
                    surf(frames_out_child(echo_idx).t_for_spect,...
                        F_for_spect,10*log10(frames_out_child(echo_idx).PowerSpectrum),...
                        "DisplayName","Power Spectrum","EdgeColor","none");
                    view([0,90]); hold on
                    axis([frames_out_child(echo_idx).t_for_spect(1) frames_out_child(echo_idx).t_for_spect(end) F_for_spect(1) F_for_spect(end)])
                    xlabel('Time (s)');hold on
                    plot(frames_out_child(echo_idx).t_for_spect,frames_out_child(echo_idx).F1,"b","DisplayName","F_1");
                    hold on;
                    plot(frames_out_child(echo_idx).t_for_spect,frames_out_child(echo_idx).F2,"r","DisplayName","F_2");
                    hold on;
                    plot(frames_out_child(echo_idx).t_for_spect,frames_out_child(echo_idx).F3,"k","DisplayName","F_3");
                    hold on;
                    plot(frames_out_child(echo_idx).t_for_spect,frames_out_child(echo_idx).pitch,"m","DisplayName","F_0");hold on;
                    
                    ylabel('Frequency (Hz)')
                    title("child")
                    pause(2)
                    soundsc(frames_out_child(echo_idx).warpedSignal,Fs);
                    figure(2);subplot(3,3,6)
                    surf(frames_out_child(echo_idx).t_for_spect,frames_out_child(echo_idx).F,...
                        (frames_out_child(echo_idx).LPC_mat),"DisplayName","LPC_{Spectrum}","EdgeColor","none");
                    view([0,90]); hold on
                    axis([frames_out_child(echo_idx).t_for_spect(1) frames_out_child(echo_idx).t_for_spect(end) frames_out_child(echo_idx).F(1) frames_out_child(echo_idx).F(end)])
                    xlabel('Time (s)');hold on
                    plot(frames_out_child(echo_idx).t_for_spect,frames_out_child(echo_idx).F1,"b","DisplayName","F_1");
                    hold on;
                    plot(frames_out_child(echo_idx).t_for_spect,frames_out_child(echo_idx).F2,"r","DisplayName","F_2");
                    hold on;
                    plot(frames_out_child(echo_idx).t_for_spect,frames_out_child(echo_idx).F3,"k","DisplayName","F_3");
                    hold on;ylabel('Frequency (Hz)');title("Child LPC")
                    
                    hold on; figure(2);subplot(3,3,8);
                    surf(frames_out_child(echo_idx).t_for_spectWarped,...
                        frames_out_child(echo_idx).FWarped,10*log10(frames_out_child(echo_idx).PowerSpectrumWarped),...
                        "DisplayName","Power Spectrum","EdgeColor","none");
                    view([0,90]); hold on
                    axis([frames_out_child(echo_idx).t_for_spectWarped(1) frames_out_child(echo_idx).t_for_spectWarped(end) frames_out_child(echo_idx).F(1) frames_out_child(echo_idx).F(end)])
                    xlabel('Time (s)');hold on
                    plot(frames_out_child(echo_idx).t_for_spectWarped,frames_out_child(echo_idx).F1Warped,"b","DisplayName","F_1");
                    hold on;
                    plot(frames_out_child(echo_idx).t_for_spectWarped,frames_out_child(echo_idx).F2Warped,"r","DisplayName","F_2");
                    hold on;
                    plot(frames_out_child(echo_idx).t_for_spectWarped,frames_out_child(echo_idx).F3Warped,"k","DisplayName","F_3");
                    hold on;
                    plot(frames_out_child(echo_idx).t_for_spectWarped,frames_out_child(echo_idx).pitchWarped,"m","DisplayName","F_0");hold on;
                    ylabel('Frequency (Hz)')
                    title("child Warped spectrum")
                    
                    
                    figure(2);subplot(3,3,9)
                    surf(frames_out_child(echo_idx).t_for_spect,frames_out_child(echo_idx).FWarped_LPC,...
                        (frames_out_child(echo_idx).LPC_matWarped),"DisplayName","LPC_{Spectrum}","EdgeColor","none");
                    view([0,90]); hold on
                    axis([frames_out_child(echo_idx).t_for_spect(1) frames_out_child(echo_idx).t_for_spect(end) frames_out_child(echo_idx).F(1) frames_out_child(echo_idx).F(end)])
                    xlabel('Time (s)');hold on
                    plot(frames_out_child(echo_idx).t_for_spect,frames_out_child(echo_idx).F1Warped,"b","DisplayName","F_1");
                    hold on;
                    plot(frames_out_child(echo_idx).t_for_spect,frames_out_child(echo_idx).F2Warped,"r","DisplayName","F_2");
                    hold on;
                    plot(frames_out_child(echo_idx).t_for_spect,frames_out_child(echo_idx).F3Warped,"k","DisplayName","F_3");
                    hold on;ylabel('Frequency (Hz)');title("child Warped spectrum LPC")
%                     c = hot(50);
%                     colormap(c);
                    
                    
                    figure(90)
                    
                    plot(frames_out_child(echo_idx).t_for_spect,frames_out_child(echo_idx).Voice,"b","DisplayName","Child Voice_{prob}");hold on;
                    hold on
                    plot(frames_out_child(echo_idx).t_for_spectWarped,frames_out_child(echo_idx).VoiceWarped,"m","DisplayName","Child Warped Voice_{prob}");hold on;
                    hold on
                    plot(frames_out_Therapist(w).t_for_spect,frames_out_Therapist(w).Voice,"k","DisplayName","Therapist  Voice_{prob}");
                    ylim([0 1]);legend();
                    
                    
                    %                     [Rmm,lags] = xcorr(frames_out_child(echo_idx).warpedSignal);
                    %
                    %                     Rmm = Rmm(lags>0);
                    %                     lags = lags(lags>0);
                    %                     figure
                    %                     plot(lags/Fs,Rmm)
                end
                
            end
        end
        
    end
    %     F1_ther = [F1_ther F1_ther(~isnan(F1_ther)) ];
    %     F2_ther = [F2_ther F2_ther(~isnan(F2_ther)) ];
    %     F3_ther = [F3_ther F3_ther(~isnan(F3_ther)) ];
    %
    %     F1_ch = [F1_ch F1_ch(~isnan(F1_ch)) ];
    %     F2_ch = [F2_ch F2_ch(~isnan(F2_ch)) ];
    %     F3_ch = [F3_ch F3_ch(~isnan(F3_ch)) ];
    %     F1_chWa = [F1_chWa F1_chWa(~isnan(F1_chWa)) ];
    %     F2_chWa = [F2_chWa F2_chWa(~isnan(F2_chWa)) ];
    %     F3_chWa = [F3_chWa F3_chWa(~isnan(F3_chWa)) ];
    %
    %     mean_F1_ther(file) = mean(F1_ther(~isnan(F1_ther)) );
    %     mean_F2_ther(file) = mean(F2_ther(~isnan(F2_ther)) );
    %     mean_F3_ther(file) = mean(F3_ther(~isnan(F3_ther)) );
    %     mean_F1_ch(file) = mean(F1_ch(~isnan(F1_ch)) );
    %     mean_F2_ch(file) = mean(F2_ch(~isnan(F2_ch)) );
    %     mean_F3_ch(file) = mean(F3_ch(~isnan(F3_ch)) );
    %     mean_F1_chWa(file) = mean(F1_chWa(~isnan(F1_chWa)) );
    %     mean_F2_chWa(file) = mean(F2_chWa(~isnan(F2_chWa)) );
    %     mean_F3_chWa(file) = mean(F3_chWa(~isnan(F3_chWa)) );
    %
end

% display the histograms of the Formants (F1-F3)

figure
histogram(F1_ther,NumBins=25,DisplayName="F1 therapist")
xline(mean(F1_ther),"DisplayName","mean F1- therapist " + num2str(round(mean(F1_ther))),'Color','b')
title([' F1 Therapist Child  ']);xlabel('Frequency[Hz]');
hold on
histogram(F1_ch,NumBins=25,DisplayName="F1 child");legend
xline(mean(F1_ch),"DisplayName","mean F1- Child "+ num2str(round(mean(F1_ch))),'Color','r')
xlabel('Frequency[Hz]');

figure
histogram(F2_ther,NumBins=25,DisplayName="F2 therapist")
xline(mean(F2_ther),"DisplayName","mean F2- therapist " + num2str(round(mean(F2_ther))),'Color','b')
title([' F2 Therapist Child  ']);xlabel('Frequency[Hz]');
hold on
histogram(F2_ch,NumBins=25,DisplayName="F2 child");legend
xline(mean(F2_ch),"DisplayName","mean F2- Child "+ num2str(round(mean(F2_ch))),'Color','r')
xlabel('Frequency[Hz]');

figure
histogram(F3_ther,NumBins=25,DisplayName="F3 therapistn")
xline(mean(F3_ther),"DisplayName","mean F3- therapist " + num2str(round(mean(F3_ther))),'Color','b')
title([' F3 Therapist Child  ']);xlabel('Frequency[Hz]');
hold on
histogram(F3_ch,NumBins=25,DisplayName="F3 child");legend
xline(mean(F3_ch),"DisplayName","mean F3- Child "+ num2str(round(mean(F3_ch))),'Color','r')
xlabel('Frequency[Hz]');




% display the histograms of the mean Formants (F1-F3)

figure
histogram(mean_F1_ther(:),NumBins=15,DisplayName="F1 therapist")
xline(mean(mean_F1_ther(:)),"DisplayName","mean F1- therapist " + num2str(round(mean(mean_F1_ther(:)))),'Color','b')
title([' F1 Therapist Child  ']);xlabel('Frequency[Hz]');
hold on
histogram(mean_F1_ch(:),NumBins=15,DisplayName="F1 child");legend
xline(mean(mean_F1_ch(:)),"DisplayName","mean F1- Child "+ num2str(round(mean(mean_F1_ch(:)))),'Color','r')
xlabel('Frequency[Hz]');

figure
histogram(mean_F2_ther(:),NumBins=15,DisplayName="F2 therapist")
xline(mean(mean_F2_ther(:)),"DisplayName","mean F2- therapist " + num2str(round(mean(mean_F2_ther(:)))),'Color','b')
title([' F2 Therapist Child  ']);xlabel('Frequency[Hz]');
hold on
histogram(mean_F2_ch(:),NumBins=15,DisplayName="F2 child");legend
xline(mean(mean_F2_ch(:)),"DisplayName","mean F2- Child "+ num2str(round(mean(mean_F2_ch(:)))),'Color','r')
xlabel('Frequency[Hz]');

figure
histogram(mean_F3_ther(:),NumBins=15,DisplayName="F3 therapistn")
xline(mean(mean_F3_ther(:)),"DisplayName","mean F3- therapist " + num2str(round(mean(mean_F3_ther(:)))),'Color','b')
title([' F3 Therapist Child  ']);xlabel('Frequency[Hz]');
hold on
histogram(mean_F3_ch(:),NumBins=15,DisplayName="F3 child");legend
xline(mean(mean_F3_ch(:)),"DisplayName","mean F3- Child "+ num2str(round(mean(mean_F3_ch(:)))),'Color','r')
xlabel('Frequency[Hz]');


%
%
% [Rmm,lags] = xcorr(frames_out_child(echo_idx).warpedSignal);
%
% Rmm = Rmm(lags>0);
% lags = lags(lags>0);
% figure
% plot(lags/Fs,Rmm)
% xlabel('Lag (s)')
% [~,dl] = findpeaks(Rmm,lags);
% %         % filtfilt
% warpedSignal2 = filter(1,[1 zeros(1,floor(dl(end))-1) 0.8],warpedSignal);
