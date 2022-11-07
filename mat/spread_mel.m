function band=spread_mel(hz_points,hz_c,hz_size,hz_max)
% Function spread_mel(hz_points,hz_c,hz_size,hz_max) in Listing 5.2
% computes the mel spreading function in this way.While
% hz_points- is a discrete array of frequencies in Hz, it should be spaced
%            in mel frequencies.
% hz_c- is defined here as the frequency index for which we are computing
%       the masking (rather than the actual frequency).
% hz_size- defines the resolution of the output masking array
% hz_max is the upper frequency limit to evaluate
% over (normally set to the Nyquist frequency).

band=zeros(1, hz_size);
hz1=hz_points(max(1,hz_c-1)); %start
hz2=hz_points(hz_c); %middle
hz3=hz_points(min(length(hz_points),hz_c+1)); %end
%-----
for hi=1:hz_size
    hz=hi*hz_max/hz_size;
    if hz > hz3
        band(hi)=0;
    elseif hz>=hz2
        if hz3-hz2==0
            band(hi)=0;
        else
            band(hi)=(hz3-hz)/(hz3-hz2);
        end
    elseif hz>=hz1
        band(hi)=(hz-hz1)/(hz2-hz1);
    else
        band(hi)=0;
    end
end

end