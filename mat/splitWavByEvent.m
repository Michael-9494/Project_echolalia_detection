function frames = splitWavByEvent(wav, EventStart,EventEnd,Fs,ADOS_table,Param)

frames = struct();
% ZCR_vec = [];
% NRG_vec = [];
for i = 1:length(EventStart)

    if length(round(EventStart(i)*Fs):round(EventEnd(i)*Fs))>= 110*10^(-3)*Fs &&...
            length(round(EventStart(i)*Fs):round(EventEnd(i)*Fs))<= 3*Fs

        frames(i).data = wav(round(EventStart(i)*Fs):round(EventEnd(i)*Fs));
        frames(i).speakreLabel = ADOS_table.Var3(ADOS_table.Var1== EventStart(i));
        frames(i).event = ADOS_table.Var4(ADOS_table.Var1== EventStart(i));
        frames(i).start_time = ADOS_table.Var1(ADOS_table.Var1== EventStart(i));

        FramedSig=enframe(frames(i).data,hamming(Param.WindowLenSamp,"periodic"),Param.noverlap);
        frames(i).ZCR = calcZCR( FramedSig);
        frames(i).NRG = calcNRG( FramedSig);

    else
        continue
    end% remain with segments in between 110 ms and 3 s
    
end