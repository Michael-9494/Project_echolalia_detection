clear;close all;clc
Data_Folder_out = "Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\";
Fs = 16000;
flag =0;
Autism_data = dir(Data_Folder_out); Autism_data = Autism_data(4:end);
Recs_for_cry_scream =[Autism_data(1).name, Autism_data(2).name, Autism_data(3).name,...
    Autism_data(4).name, Autism_data(5).name, Autism_data(6).name,...
    Autism_data(7).name, Autism_data(8).name, Autism_data(9).name];
Recs_for_cry_scream1 = split(Recs_for_cry_scream,"Recs_for_cry_scream");
Recs_for_cry_scream1 = Recs_for_cry_scream1(2:end);


for i = 1:length(Recs_for_cry_scream)
        try
            load("Recs_for_cry_scream"+Recs_for_cry_scream1(i)+".mat")
        catch ME
            fprintf('load without success: %s\n', ME.message);
            continue;  % Jump to next iteration of: for i
        end
    
    % Recs_for_cry_scream_18092022_ECHO
    for file = 1:length(all)

        len_ther_struct = length(all(file).frames_out_Therapist);
        len_child_struct = length(all(file).frames_out_child);
        if len_ther_struct > len_child_struct

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
                        if flag ==1


                            pause(2)
                            soundsc(all(file).frames_out_Therapist(w).Signal_frame,16000);
                            pause(2)
                            soundsc(all(file).frames_out_child(echo_idx).Signal_frame,16000);


                            figure
                            subplot(2,2,1)
                            plot(all(file).frames_out_Therapist(w).t,all(file).frames_out_Therapist(w).Signal_frame)
                            title([' Therapist']);
                            xlabel('time[s]');ylabel('amp'); axis tight

                            hold on; subplot(2,2,3); plot(all(file).frames_out_child(echo_idx).t,...
                                all(file).frames_out_child(echo_idx).Signal_frame)
                            title('Child ' );
                            xlabel('time[s]');ylabel('amp')
                            axis tight; hold on


                            subplot(2,2,2)
                            surf(all(file).frames_out_Therapist(w).t_for_spect,all(file).frames_out_Therapist(w).F,...
                                10*log10(all(file).frames_out_Therapist(w).PowerSpectrum),...
                                "DisplayName","P_{Spectrum}","EdgeColor","none");
                            view([0,90]); hold on
                            axis([all(file).frames_out_Therapist(w).t_for_spect(1) all(file).frames_out_Therapist(w).t_for_spect(end) all(file).frames_out_Therapist(w).F(1) all(file).frames_out_Therapist(w).F(end)])
                            xlabel('Time (s)');hold on
                            plot(all(file).frames_out_Therapist(w).t_for_spect,...
                                all(file).frames_out_Therapist(w).F1,"b","DisplayName","F_1",'LineWidth',1.5);
                            hold on;
                            plot(all(file).frames_out_Therapist(w).t_for_spect,...
                                all(file).frames_out_Therapist(w).F2,"r","DisplayName","F_2",'LineWidth',1.5);
                            hold on;
                            plot(all(file).frames_out_Therapist(w).t_for_spect,...
                                all(file).frames_out_Therapist(w).F3,"k","DisplayName","F_3",'LineWidth',1.5);
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
                                all(file).frames_out_child(echo_idx).F1,"b","DisplayName","F_1",'LineWidth',1.5);
                            hold on;
                            plot(all(file).frames_out_child(echo_idx).t_for_spect,...
                                all(file).frames_out_child(echo_idx).F2,"r","DisplayName","F_2",'LineWidth',1.5);
                            hold on;
                            plot(all(file).frames_out_child(echo_idx).t_for_spect,...
                                all(file).frames_out_child(echo_idx).F3,"k","DisplayName","F_3",'LineWidth',1.5);
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
            end % end for: w = 1:length(all(file).frames_out_Therapist)

        else
            for w = 1:length(all(file).frames_out_child)
                if isempty(all(file).frames_out_child(w).Signal_frame)
                    continue
                else

                    if ~strcmp( all(file).frames_out_child(w).segment_event ,'Echolalia')
                        continue
                    else
                        % find the closest echo segment of child
                        for j = 1:length(all(file).frames_out_Therapist)
                            if isempty(all(file).frames_out_Therapist(j).Signal_frame)
                                continue
                            else
                                if all(file).frames_out_Therapist(j).t(end) <(all(file).frames_out_child(w).t(1) + 2) &&...
                                        strcmp( all(file).frames_out_Therapist(j).segment_event ,'Echolalia')

                                    echo_idx = j;
                                    break
                                end
                            end
                        end
                        if flag ==1


                            pause(2)
                            soundsc(all(file).frames_out_Therapist(w).Signal_frame,16000);
                            pause(2)
                            soundsc(all(file).frames_out_child(echo_idx).Signal_frame,16000);


                            figure
                            subplot(2,2,1)
                            plot(all(file).frames_out_Therapist(w).t,all(file).frames_out_Therapist(w).Signal_frame)
                            title([' Therapist']);
                            xlabel('time[s]');ylabel('amp'); axis tight

                            hold on; subplot(2,2,3); plot(all(file).frames_out_child(echo_idx).t,...
                                all(file).frames_out_child(echo_idx).Signal_frame)
                            title('Child ' );
                            xlabel('time[s]');ylabel('amp')
                            axis tight; hold on


                            subplot(2,2,2)
                            surf(all(file).frames_out_Therapist(w).t_for_spect,all(file).frames_out_Therapist(w).F,...
                                10*log10(all(file).frames_out_Therapist(w).PowerSpectrum),...
                                "DisplayName","P_{Spectrum}","EdgeColor","none");
                            view([0,90]); hold on
                            axis([all(file).frames_out_Therapist(w).t_for_spect(1) all(file).frames_out_Therapist(w).t_for_spect(end) all(file).frames_out_Therapist(w).F(1) all(file).frames_out_Therapist(w).F(end)])
                            xlabel('Time (s)');hold on
                            plot(all(file).frames_out_Therapist(w).t_for_spect,...
                                all(file).frames_out_Therapist(w).F1,"b","DisplayName","F_1",'LineWidth',1.5);
                            hold on;
                            plot(all(file).frames_out_Therapist(w).t_for_spect,...
                                all(file).frames_out_Therapist(w).F2,"r","DisplayName","F_2",'LineWidth',1.5);
                            hold on;
                            plot(all(file).frames_out_Therapist(w).t_for_spect,...
                                all(file).frames_out_Therapist(w).F3,"k","DisplayName","F_3",'LineWidth',1.5);
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
                                all(file).frames_out_child(echo_idx).F1,"b","DisplayName","F_1",'LineWidth',1.5);
                            hold on;
                            plot(all(file).frames_out_child(echo_idx).t_for_spect,...
                                all(file).frames_out_child(echo_idx).F2,"r","DisplayName","F_2",'LineWidth',1.5);
                            hold on;
                            plot(all(file).frames_out_child(echo_idx).t_for_spect,...
                                all(file).frames_out_child(echo_idx).F3,"k","DisplayName","F_3",'LineWidth',1.5);
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
            end % end for: w = 1:length(all(file).frames_out_Therapist)

        end % end  if :len_ther_struct > len_child_struct

        mean_F1_ther(file,i) = mean(all(file).F1_ther(~isnan(all(file).F1_ther)) );
        mean_F2_ther(file,i) = mean(all(file).F2_ther(~isnan(all(file).F2_ther)) );
        mean_F3_ther(file,i) = mean(all(file).F3_ther(~isnan(all(file).F3_ther)) );
        mean_F1_ch(file,i) = mean(all(file).F1_ch(~isnan(all(file).F1_ch)) );
        mean_F2_ch(file,i) = mean(all(file).F2_ch(~isnan(all(file).F2_ch)) );
        mean_F3_ch(file,i) = mean(all(file).F3_ch(~isnan(all(file).F3_ch)) );

    end
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

