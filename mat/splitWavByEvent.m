function frames = splitWavByEvent(wav, EventStart,EventEnd,Fs,ADOS_table,Param)

frames = struct();
ZCR_vec = [];
NRG_vec = [];
for i = 1:length(EventStart)

    if length(round(EventStart(i)*Fs):round(EventEnd(i)*Fs))>= 110*10^(-3)*Fs &&...
            length(round(EventStart(i)*Fs):round(EventEnd(i)*Fs))<= 3*Fs

        frames(i).data = wav(round(EventStart(i)*Fs):round(EventEnd(i)*Fs));
        frames(i).speakreLabel = ADOS_table.Var3(ADOS_table.Var1== EventStart(i));
        frames(i).event = ADOS_table.Var4(ADOS_table.Var1== EventStart(i));
        frames(i).start_time = ADOS_table.Var1(ADOS_table.Var1== EventStart(i));

%         [~,FramedSig]=PreProcess(frames(i).data,Fs,Param.alpha,Param.WindowLength,Param.Overlap);
%         frames(i).ZCR = calcZCR( FramedSig)';
%         frames(i).NRG = calcNRG( FramedSig)';
%         ZCR_vec = [ZCR_vec frames(i).ZCR];
%         NRG_vec = [NRG_vec frames(i).NRG];
        %         mzcr = mean(ZeroCrossingSignal);
        %         vzcr = var(ZeroCrossingSignal,1);
        %         EbNRG = mean(EnergySignal);
        %         frames(i).EbZCR = mzcr-(sqrt(vzcr)/100);
        %         frames(i).Eb_NRG = 10 * log10(1.3) + EbNRG;
    else
        continue
    end
    % remain with segments in between 110 ms and 3 s
% 
%     frames.ZCR_vec = ZCR_vec;
%     frames.NRG_vec = NRG_vec;
end