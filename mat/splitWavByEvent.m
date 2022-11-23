function frames = splitWavByEvent(wav, EventStart,EventEnd,Fs,ADOS_table)

frames = struct();
for i = 1:length(EventStart)
	frames(i).data = wav(round(EventStart(i)*Fs):round(EventEnd(i)*Fs));
    frames(i).speakreLabel = ADOS_table.Var3(ADOS_table.Var1== EventStart(i));
    frames(i).event = ADOS_table.Var4(ADOS_table.Var1== EventStart(i));
    frames(i).start_time = ADOS_table.Var1(ADOS_table.Var1== EventStart(i));
end