function frames = splitWavByEvent(wav, EventStart,EventEnd,Fs,ADOS_table,Param)

frames = struct();
for i = 1:length(EventStart)
    %     [~,FramedSig]=PreProcess(wav(round((EventStart(i)-20)*Fs):round((EventEnd(i)+20)*Fs)),...
    %         Fs,Param.alpha,Param.WindowLength,Param.Overlap);
    %
    %     ZeroCrossingSignal = calcZCR( FramedSig);
    %     EnergySignal=calcNRG(FramedSig);
    %     mzcr = mean(ZeroCrossingSignal);
    %     vzcr = var(ZeroCrossingSignal,1);
    %     EbNRG = mean(EnergySignal);
    %     frames(i).EbZCR = mzcr-(sqrt(vzcr)/100);
    %     frames(i).Eb_NRG = 10 * log10(1.3) + EbNRG;
    if length(round(EventStart(i)*Fs):round(EventEnd(i)*Fs))>= 110*10^(-3)*Fs &&...
            length(round(EventStart(i)*Fs):round(EventEnd(i)*Fs))<= 3*Fs

        frames(i).data = wav(round(EventStart(i)*Fs):round(EventEnd(i)*Fs));
        frames(i).speakreLabel = ADOS_table.Var3(ADOS_table.Var1== EventStart(i));
        frames(i).event = ADOS_table.Var4(ADOS_table.Var1== EventStart(i));
        frames(i).start_time = ADOS_table.Var1(ADOS_table.Var1== EventStart(i));
    else
        continue
    end
    % remain with segments in between 110 ms and 3 s


end