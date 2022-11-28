clear;close all;clc
Data_Folder = "Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\Recs_for_cry_scream_18092022\";
Fs = 16000;
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
                        if all(file).frames_out_Therapist(w).t(1) <(all(file).frames_out_child(j).t(1) + 5) &&...
                                strcmp( all(file).frames_out_child(j).segment_event ,'Echolalia')
                            echo_idx = j;
                            break
                        end
                    end
                end
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







