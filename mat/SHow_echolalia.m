clear;close all;clc
Data_Folder_out = "Y:\Database\Recordings_for_Segmentor\Michael\Recs_for_cry_scream\";
Fs = 16000;
flag =0;
Autism_data = dir(Data_Folder_out); Autism_data = Autism_data(4:end);
% Recs_for_cry_scream =[Autism_data(1).name, Autism_data(2).name, Autism_data(3).name,...
%     Autism_data(4).name, Autism_data(5).name, Autism_data(6).name,...
%     Autism_data(7).name, Autism_data(8).name, Autism_data(9).name];
% Recs_for_cry_scream1 = split(Recs_for_cry_scream,"Recs_for_cry_scream");
Recs_for_cry_scream1 ="_25092022";

F1_ther = [];F2_ther = [];F3_ther = [];
F1_ch = [];F2_ch = [];F3_ch = [];

for i = 1:length(Recs_for_cry_scream1)
    echo_count =0;echo_count1 =0;
    try
        load("Recs_for_cry_scream1"+Recs_for_cry_scream1(i)+".mat")
    catch ME
        fprintf('load without success: %s\n', ME.message);
        continue;  % Jump to next iteration of: for i
    end

    % Recs_for_cry_scream_18092022_ECHO
    for file = 1:length(all)
        % if the current row is empty
        if isempty(all(file).Child_name)
            continue % Jump to next iteration of: for file
        else
            len_ther_struct = length(all(file).frames_out_Therapist);
            len_child_struct = length(all(file).frames_out_child);
            if len_ther_struct > len_child_struct
                
                for w = 1:length(all(file).frames_out_Therapist)
                    % if we do not have data-> continue
                    if isempty(all(file).frames_out_Therapist(w).Signal_frame)
                        continue
                    else
                        %                         if w length(all(file).frames_out_Therapist)
                        % if the segment is not echolalia -> continue w +1
                        if ~strcmp( all(file).frames_out_Therapist(w).segment_event ,'Echolalia')
                            continue
                        else
                            echo_count =echo_count+1;
                            % find the closest echo segment of child
                            echo_idx = 0;
                            for j = 1:length(all(file).frames_out_child)
                                if isempty(all(file).frames_out_child(j).Signal_frame)
                                    continue
                                else
                                    if (all(file).frames_out_Therapist(w).t(end)+15) >(all(file).frames_out_child(j).t(1)) &&...
                                            ((all(file).frames_out_Therapist(w).t(end)) <=(all(file).frames_out_child(j).t(1)))  &&...
                                            strcmp( all(file).frames_out_child(j).segment_event{1} ,'Echolalia')
                                        % assine new echolalia of child and exit the for loop
                                        echo_idx = j;
                                        break
                                    end
                                end
                            end
                            % move to print
                            if flag ==1 && echo_idx~=0


                                pause(2)
                                soundsc(all(file).frames_out_Therapist(w).Signal_frame,16000);
                                pause(2)
                                soundsc(all(file).frames_out_child(echo_idx).Signal_frame,16000);


                                figure
                                subplot(2,2,1)
                                plot(all(file).frames_out_Therapist(w).t,all(file).frames_out_Therapist(w).Signal_frame)
                                title([' Therapist ' + string(all(file).frames_out_Therapist(w).segment_event{1})]);
                                xlabel('time[s]');ylabel('amp'); axis tight

                                hold on; subplot(2,2,3); plot(all(file).frames_out_child(echo_idx).t,...
                                    all(file).frames_out_child(echo_idx).Signal_frame)
                                title(['Child '  + string(all(file).frames_out_child(echo_idx).segment_event{1})]);
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
                            echo_count1 = echo_count1 +1;
                        else
                            echo_idx = 0;
                            % find the closest echo segment of child
                            for j = 1:length(all(file).frames_out_Therapist)
                                if isempty(all(file).frames_out_Therapist(j).Signal_frame)
                                    continue
                                else
                                    if (all(file).frames_out_Therapist(j).t(end)+15) >(all(file).frames_out_child(w).t(1)) &&...
                                            ((all(file).frames_out_Therapist(j).t(end)) <=(all(file).frames_out_child(w).t(1)))  &&...
                                            strcmp( all(file).frames_out_Therapist(j).segment_event{1} ,'Echolalia')

                                        echo_idx = j;
                                        break
                                    end
                                end
                            end
                            if flag ==1 && echo_idx~=0


                                pause(2)
                                soundsc(all(file).frames_out_Therapist(echo_idx).Signal_frame,16000);
                                pause(2)
                                soundsc(all(file).frames_out_child(w).Signal_frame,16000);


                                figure
                                subplot(2,2,1)
                                plot(all(file).frames_out_Therapist(echo_idx).t,all(file).frames_out_Therapist(echo_idx).Signal_frame)
                                title([' Therapist ' + string(all(file).frames_out_Therapist(echo_idx).segment_event{1})]);
                                xlabel('time[s]');ylabel('amp'); axis tight

                                hold on; subplot(2,2,3); plot(all(file).frames_out_child(w).t,...
                                    all(file).frames_out_child(w).Signal_frame)
                                title(['Child '  + string(all(file).frames_out_child(w).segment_event{1})]);
                                xlabel('time[s]');ylabel('amp')
                                axis tight; hold on

                                subplot(2,2,2)
                                surf(all(file).frames_out_Therapist(echo_idx).t_for_spect,all(file).frames_out_Therapist(echo_idx).F,...
                                    10*log10(all(file).frames_out_Therapist(echo_idx).PowerSpectrum),...
                                    "DisplayName","P_{Spectrum}","EdgeColor","none");
                                view([0,90]); hold on
                                axis([all(file).frames_out_Therapist(echo_idx).t_for_spect(1) all(file).frames_out_Therapist(echo_idx).t_for_spect(end) all(file).frames_out_Therapist(echo_idx).F(1) all(file).frames_out_Therapist(echo_idx).F(end)])
                                xlabel('Time (s)');hold on
                                plot(all(file).frames_out_Therapist(echo_idx).t_for_spect,...
                                    all(file).frames_out_Therapist(echo_idx).F1,"b","DisplayName","F_1",'LineWidth',1.5);
                                hold on;
                                plot(all(file).frames_out_Therapist(echo_idx).t_for_spect,...
                                    all(file).frames_out_Therapist(echo_idx).F2,"r","DisplayName","F_2",'LineWidth',1.5);
                                hold on;
                                plot(all(file).frames_out_Therapist(echo_idx).t_for_spect,...
                                    all(file).frames_out_Therapist(echo_idx).F3,"k","DisplayName","F_3",'LineWidth',1.5);
                                hold on;
                                %     plot(all(file).frames_out_Therapist(w).t_for_spect,...
                                %         all(file).frames_out_Therapist(w).F4,"g","DisplayName","F_4");
                                ylabel('Frequency (Hz)')
                                %     c = colorbar;
                                %     c.Label.String = 'Power (dB)';
                                title("Therapist")
                                %     legend

                                subplot(2,2,4)
                                surf(all(file).frames_out_child(w).t_for_spect,...
                                    all(file).frames_out_child(w).F,10*log10(all(file).frames_out_child(w).PowerSpectrum),...
                                    "DisplayName","Power Spectrum","EdgeColor","none");
                                view([0,90]); hold on
                                axis([all(file).frames_out_child(w).t_for_spect(1) all(file).frames_out_child(w).t_for_spect(end) all(file).frames_out_child(w).F(1) all(file).frames_out_child(w).F(end)])
                                xlabel('Time (s)');hold on
                                plot(all(file).frames_out_child(w).t_for_spect,...
                                    all(file).frames_out_child(w).F1,"b","DisplayName","F_1",'LineWidth',1.5);
                                hold on;
                                plot(all(file).frames_out_child(w).t_for_spect,...
                                    all(file).frames_out_child(w).F2,"r","DisplayName","F_2",'LineWidth',1.5);
                                hold on;
                                plot(all(file).frames_out_child(w).t_for_spect,...
                                    all(file).frames_out_child(w).F3,"k","DisplayName","F_3",'LineWidth',1.5);
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

                end % end for: w = 1:length(all(file).frames_out_child)

            end % end  if :len_ther_struct > len_child_struct

            F1_ther = [F1_ther all(file).F1_ther(~isnan(all(file).F1_ther)) ];
            F2_ther = [F2_ther all(file).F2_ther(~isnan(all(file).F2_ther)) ];
            F3_ther = [F3_ther all(file).F3_ther(~isnan(all(file).F3_ther)) ];
            F1_ch = [F1_ch all(file).F1_ch(~isnan(all(file).F1_ch)) ];
            F2_ch = [F2_ch all(file).F2_ch(~isnan(all(file).F2_ch)) ];
            F3_ch = [F3_ch all(file).F3_ch(~isnan(all(file).F3_ch)) ];

            mean_F1_ther(file,i) = mean(all(file).F1_ther(~isnan(all(file).F1_ther)) );
            mean_F2_ther(file,i) = mean(all(file).F2_ther(~isnan(all(file).F2_ther)) );
            mean_F3_ther(file,i) = mean(all(file).F3_ther(~isnan(all(file).F3_ther)) );
            mean_F1_ch(file,i) = mean(all(file).F1_ch(~isnan(all(file).F1_ch)) );
            mean_F2_ch(file,i) = mean(all(file).F2_ch(~isnan(all(file).F2_ch)) );
            mean_F3_ch(file,i) = mean(all(file).F3_ch(~isnan(all(file).F3_ch)) );
        end
    end
end
% get rid of zeros
mean_F1_ther = nonzeros( mean_F1_ther);
mean_F2_ther = nonzeros( mean_F2_ther);
mean_F3_ther = nonzeros( mean_F3_ther);
mean_F1_ch = nonzeros( mean_F1_ch);
mean_F2_ch = nonzeros( mean_F2_ch);
mean_F3_ch = nonzeros( mean_F3_ch);


% display the histograms of the Formants (F1-F3)

figure
histogram(F1_ther,NumBins=25,DisplayName="F1 therapist")
xline(mean(F1_ther),"DisplayName","mean F1- therapist " + num2str(round(mean(F1_ther))),'Color','b')
title(' F1 Therapist Child  ');xlabel('Frequency[Hz]');
hold on
histogram(F1_ch,NumBins=25,DisplayName="F1 child");legend
xline(mean(F1_ch),"DisplayName","mean F1- Child "+ num2str(round(mean(F1_ch))),'Color','r')
xlabel('Frequency[Hz]');

figure
histogram(F2_ther,NumBins=25,DisplayName="F2 therapist")
xline(mean(F2_ther),"DisplayName","mean F2- therapist " + num2str(round(mean(F2_ther))),'Color','b')
title(' F2 Therapist Child  ');xlabel('Frequency[Hz]');
hold on
histogram(F2_ch,NumBins=25,DisplayName="F2 child");legend
xline(mean(F2_ch),"DisplayName","mean F2- Child "+ num2str(round(mean(F2_ch))),'Color','r')
xlabel('Frequency[Hz]');

figure
histogram(F3_ther,NumBins=25,DisplayName="F3 therapistn")
xline(mean(F3_ther),"DisplayName","mean F3- therapist " + num2str(round(mean(F3_ther))),'Color','b')
title(' F3 Therapist Child  ');xlabel('Frequency[Hz]');
hold on
histogram(F3_ch,NumBins=25,DisplayName="F3 child");legend
xline(mean(F3_ch),"DisplayName","mean F3- Child "+ num2str(round(mean(F3_ch))),'Color','r')
xlabel('Frequency[Hz]');




% display the histograms of the mean Formants (F1-F3)

figure
histogram(mean_F1_ther(:),NumBins=15,DisplayName="F1 therapist")
xline(mean(mean_F1_ther(:)),"DisplayName","mean F1- therapist " + num2str(round(mean(mean_F1_ther(:)))),'Color','b')
title(' F1 Therapist Child  ');xlabel('Frequency[Hz]');
hold on
histogram(mean_F1_ch(:),NumBins=15,DisplayName="F1 child");legend
xline(mean(mean_F1_ch(:)),"DisplayName","mean F1- Child "+ num2str(round(mean(mean_F1_ch(:)))),'Color','r')
xlabel('Frequency[Hz]');

figure
histogram(mean_F2_ther(:),NumBins=15,DisplayName="F2 therapist")
xline(mean(mean_F2_ther(:)),"DisplayName","mean F2- therapist " + num2str(round(mean(mean_F2_ther(:)))),'Color','b')
title(' F2 Therapist Child  ');xlabel('Frequency[Hz]');
hold on
histogram(mean_F2_ch(:),NumBins=15,DisplayName="F2 child");legend
xline(mean(mean_F2_ch(:)),"DisplayName","mean F2- Child "+ num2str(round(mean(mean_F2_ch(:)))),'Color','r')
xlabel('Frequency[Hz]');

figure
histogram(mean_F3_ther(:),NumBins=15,DisplayName="F3 therapistn")
xline(mean(mean_F3_ther(:)),"DisplayName","mean F3- therapist " + num2str(round(mean(mean_F3_ther(:)))),'Color','b')
title(' F3 Therapist Child  ');xlabel('Frequency[Hz]');
hold on
histogram(mean_F3_ch(:),NumBins=15,DisplayName="F3 child");legend
xline(mean(mean_F3_ch(:)),"DisplayName","mean F3- Child "+ num2str(round(mean(mean_F3_ch(:)))),'Color','r')
xlabel('Frequency[Hz]');

