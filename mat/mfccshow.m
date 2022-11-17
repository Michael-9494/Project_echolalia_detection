function mfccshow(audio_mfcc,number_samples,sampling_frequency,xtick_step)
            % mfccshow Display MFCCs in seconds.
            %   mfccshow(audio_mfcc,number_samples,sampling_frequency,xtick_step)
            %   
            %   Inputs:
            %       audio_mfcc: audio MFCCs [number_coefficients, number_times]
            %       number_samples: number of samples from the original signal
            %       sampling_frequency: sampling frequency from the original signal in Hz
            %       xtick_step: step for the x-axis ticks in seconds (default: 1 second)
            
            % Set the default values for xtick_step
            if nargin < 4
                xtick_step = 1;
            end
            
            % Get the number of time frames
            number_times = size(audio_mfcc,2);
            
            % Derive the number of seconds and the number of time frames per second
            number_seconds = number_samples/sampling_frequency;
            time_resolution = number_times/number_seconds;
            
            % Prepare the tick locations and labels for the x-axis
            xtick_locations = xtick_step*time_resolution:xtick_step*time_resolution:number_times;
            xtick_labels = xtick_step:xtick_step:number_seconds;
            
            % Display the MFCCs in seconds
            imagesc(audio_mfcc)
            axis xy
            colormap(jet)
%             c = colorbar;
%             c.Label.String = 'Power (dB)';

            xticks(xtick_locations)
            xticklabels(xtick_labels)
            xlabel('Time (s)')
            ylabel('Coefficients')
            
        end