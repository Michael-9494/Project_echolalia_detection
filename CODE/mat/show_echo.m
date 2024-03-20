clear;close all;clc
Data_Folder = "Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\Recs_for_cry_scream_18092022\";

load("Recs_for_cry_scream_25092022.mat")
% Recs_for_cry_scream_18092022_ECHO
for file = 1:length(all)
    for w = 1:length(all(file).frames_out_Therapist)
        if isempty(all(file).frames_out_Therapist(w).Signal_frame)
            continue
        else

            if ~strcmp( all(file).frames_out_Therapist(w).segment_event ,'Echolalia')
                %                     all(file).frames_out_child(w).segment_event ~= 'Echolalia'
                continue
            else

                soundsc(all(file).frames_out_Therapist(w).Signal_frame,16000);
                mzcr_C = mean(all(file).frames_out_child(w).ZeroCrossingSignal);
                vzcr_C = var(all(file).frames_out_child(w).ZeroCrossingSignal,1);
                mnrg_C = mean(all(file).frames_out_child(w).Signal_Energy);
                vnrg_C = var(all(file).frames_out_child(w).Signal_Energy,1);
                % baseline energy level, Eb
                EbNRG_C = mnrg_C+(sqrt(vnrg_C));%
                EbZCR_C = mzcr_C-(sqrt(vzcr_C)/10);
                mzcr_T = mean(all(file).frames_out_Therapist(w).ZeroCrossingSignal);
                vzcr_T = var(all(file).frames_out_Therapist(w).ZeroCrossingSignal,1);
                mnrg_T = mean(all(file).frames_out_Therapist(w).Signal_Energy);
                vnrg_T = var(all(file).frames_out_Therapist(w).Signal_Energy,1);
                % baseline energy level, Eb
                EbNRG_T = mnrg_T+(sqrt(vnrg_T));
                EbZCR_T = mzcr_T-(sqrt(vzcr_T)/10);

                figure
                subplot(4,2,1)
                plot(all(file).frames_out_Therapist(w).t,all(file).frames_out_Therapist(w).Signal_frame)
                title('speech signal therapist');xlabel('time[s]');ylabel('amp')
                axis tight
                hold on; subplot(4,2,2); plot(all(file).frames_out_child(w).t,...
                    all(file).frames_out_child(w).Signal_frame)
                title('speech signal child');xlabel('time[s]');ylabel('amp')
                axis tight; hold on

                subplot(4,2,4),plot(1:length(all(file).frames_out_child(w).Signal_Energy),...
                    all(file).frames_out_child(w).Signal_Energy);yline(EbNRG_C,'--m');%yline(th1_C,'b');
                title("Signal Energy C"); xlabel 'segment index' ; ylabel 'Energy [dB]'
                xlim([0 length(all(file).frames_out_child(w).Signal_Energy)])
                grid on; hold on
                subplot(4,2,3),plot(1:length(all(file).frames_out_Therapist(w).Signal_Energy),...
                    all(file).frames_out_Therapist(w).Signal_Energy);yline(EbNRG_T,'--m');%yline(th1_T,'b');
                title("Signal Energy T"); xlabel 'segment index' ; ylabel 'Energy [dB]';
                xlim([0 length(all(file).frames_out_Therapist(w).Signal_Energy)])
                grid on; hold on

                subplot(4,2,6)
                plot(1:length(all(file).frames_out_child(w).ZeroCrossingSignal),all(file).frames_out_child(w).ZeroCrossingSignal),xlabel('frame number');
                ylabel('Zero-crossing rate'),title("Signal ZCR C");yline(EbZCR_C,'--m');
                xlim([0 length(all(file).frames_out_child(w).ZeroCrossingSignal)])
                subplot(4,2,5)
                plot(1:length(all(file).frames_out_Therapist(w).ZeroCrossingSignal),all(file).frames_out_Therapist(w).ZeroCrossingSignal),xlabel('frame number');
                ylabel('Zero-crossing rate'),title("Signal ZCR T");yline(EbZCR_T,'--m');
                xlim([0 length(all(file).frames_out_Therapist(w).ZeroCrossingSignal)])


                subplot(4,2,7)
                surf(all(file).frames_out_Therapist(w).t_for_spect,all(file).frames_out_Therapist(w).F,...
                    10*log10(all(file).frames_out_Therapist(w).PowerSpectrum),...
                    "DisplayName","P_{Spectrum}","EdgeColor","none");
                view([0,90]); hold on
                axis([all(file).frames_out_Therapist(w).t_for_spect(1) all(file).frames_out_Therapist(w).t_for_spect(end) all(file).frames_out_Therapist(w).F(1) all(file).frames_out_Therapist(w).F(end)])
                xlabel('Time (s)');hold on
                plot(all(file).frames_out_Therapist(w).t_for_spect,...
                    all(file).frames_out_Therapist(w).F1,"b","DisplayName","F_1");
                hold on;
                plot(all(file).frames_out_Therapist(w).t_for_spect,...
                    all(file).frames_out_Therapist(w).F2,"r","DisplayName","F_2");
                hold on;
                plot(all(file).frames_out_Therapist(w).t_for_spect,...
                    all(file).frames_out_Therapist(w).F3,"k","DisplayName","F_3");
                hold on;
                %     plot(all(file).frames_out_Therapist(w).t_for_spect,...
                %         all(file).frames_out_Therapist(w).F4,"g","DisplayName","F_4");
                ylabel('Frequency (Hz)')
                %     c = colorbar;
                %     c.Label.String = 'Power (dB)';
                title("Therapist")
                %     legend


                subplot(4,2,8)
                surf(all(file).frames_out_child(w).t_for_spect,...
                    all(file).frames_out_child(w).F,10*log10(all(file).frames_out_child(w).PowerSpectrum),...
                    "DisplayName","Power Spectrum","EdgeColor","none");
                view([0,90]); hold on
                axis([all(file).frames_out_child(w).t_for_spect(1) all(file).frames_out_child(w).t_for_spect(end) all(file).frames_out_child(w).F(1) all(file).frames_out_child(w).F(end)])
                xlabel('Time (s)');hold on
                plot(all(file).frames_out_child(w).t_for_spect,...
                    all(file).frames_out_child(w).F1,"b","DisplayName","F_1");
                hold on;
                plot(all(file).frames_out_child(w).t_for_spect,...
                    all(file).frames_out_child(w).F2,"r","DisplayName","F_2");
                hold on;
                plot(all(file).frames_out_child(w).t_for_spect,...
                    all(file).frames_out_child(w).F3,"k","DisplayName","F_3");
                hold on;
                %     plot(all(file).frames_out_child(w).t_for_spect,...
                %         all(file).frames_out_child(w).F4,"g","DisplayName","F_4");
                ylabel('Frequency (Hz)')
                %     c = colorbar;
                %     c.Label.String = 'Power (dB)';
                title("child")
                %     legend
                %     soundsc(all(file).frames_out_Therapist(w).Signal_frame,16000);
                pause(1)
                soundsc(all(file).frames_out_child(w).Signal_frame,16000);
            end
        end
    end

end



