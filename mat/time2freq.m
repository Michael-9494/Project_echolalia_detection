function freqs = time2freq(time)

freqs = struct();
for i = 1:length(time)
    freq = fft(time(i).Signal_frame);
    freqs(i).data = freq(1:ceil((length(freq)+1)/2));
end