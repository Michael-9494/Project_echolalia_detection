clear;close all;clc
Fs = 16000;
flag = 0;


load("Recs_for_cry_scream_25092022.mat")

% Recs_for_cry_scream_18092022_ECHO
for file = 1:length(all)
    for w = 1:length(all(file).frames_out_Therapist)
        if isempty(all(file).frames_out_Therapist(w).Signal_frame)
            continue
        else

            if ~strcmp( all(file).frames_out_Therapist(w).segment_event ,'Echolalia')
                continue
            else
                % find the closest echo segment of child
                for j = 1:length(all(file).frames_out_child)
                    if isempty(all(file).frames_out_child(j).Signal_frame)
                        continue
                    else
                        if all(file).frames_out_Therapist(w).t(end) <(all(file).frames_out_child(j).t(1) + 2) &&...
                                strcmp( all(file).frames_out_child(j).segment_event ,'Echolalia')
                            echo_idx = j;
                            break
                        end
                    end
                end


                if flag == 1
                    soundsc(all(file).frames_out_Therapist(w).Signal_frame,16000);
                    pause(2)
                    soundsc(all(file).frames_out_child(echo_idx).Signal_frame,16000);
                    figure
                    subplot(2,2,1)
                    plot(all(file).frames_out_Therapist(w).t,all(file).frames_out_Therapist(w).Signal_frame)
                    title(['speech signal ' all(file).frames_out_Therapist(w).segment_event]);
                    xlabel('time[s]');ylabel('amp'); axis tight

                    hold on; subplot(2,2,2); plot(all(file).frames_out_child(echo_idx).t,...
                        all(file).frames_out_child(echo_idx).Signal_frame)
                    title(['speech signal ' all(file).frames_out_child(echo_idx).segment_event]);
                    xlabel('time[s]');ylabel('amp')
                    axis tight; hold on


                    subplot(2,2,3)
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

                    subplot(2,2,4)
                    surf(all(file).frames_out_child(echo_idx).t_for_spect,...
                        all(file).frames_out_child(echo_idx).F,10*log10(all(file).frames_out_child(echo_idx).PowerSpectrum),...
                        "DisplayName","Power Spectrum","EdgeColor","none");
                    view([0,90]); hold on
                    axis([all(file).frames_out_child(echo_idx).t_for_spect(1) all(file).frames_out_child(echo_idx).t_for_spect(end) all(file).frames_out_child(echo_idx).F(1) all(file).frames_out_child(echo_idx).F(end)])
                    xlabel('Time (s)');hold on
                    plot(all(file).frames_out_child(echo_idx).t_for_spect,...
                        all(file).frames_out_child(echo_idx).F1,"b","DisplayName","F_1");
                    hold on;
                    plot(all(file).frames_out_child(echo_idx).t_for_spect,...
                        all(file).frames_out_child(echo_idx).F2,"r","DisplayName","F_2");
                    hold on;
                    plot(all(file).frames_out_child(echo_idx).t_for_spect,...
                        all(file).frames_out_child(echo_idx).F3,"k","DisplayName","F_3");
                    hold on;
                    %     plot(all(file).frames_out_child(w).t_for_spect,...
                    %         all(file).frames_out_child(w).F4,"g","DisplayName","F_4");
                    ylabel('Frequency (Hz)')
                    %     c = colorbar;
                    %     c.Label.String = 'Power (dB)';
                    title("child")

                    %     legend
                    %     soundsc(all(file).frames_out_Therapist(w).Signal_frame,16000);
                end

            end
        end
    end
    mean_F1_ther(file) = mean(all(file).F1_ther(~isnan(all(file).F1_ther)) );
    mean_F2_ther(file) = mean(all(file).F2_ther(~isnan(all(file).F2_ther)) );
    mean_F3_ther(file) = mean(all(file).F3_ther(~isnan(all(file).F3_ther)) );
    mean_F1_ch(file) = mean(all(file).F1_ch(~isnan(all(file).F1_ch)) );
    mean_F2_ch(file) = mean(all(file).F2_ch(~isnan(all(file).F2_ch)) );
    mean_F3_ch(file) = mean(all(file).F3_ch(~isnan(all(file).F3_ch)) );
end


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
