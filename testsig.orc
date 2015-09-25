
sr = 44100
kr = 441
ksmps = 100
nchnls = 4
0dbfs = 1.0



FLcolor	180,200,199
FLpanel 	"FREQGEN",200,100
            
gkfreq1,    iknob1 FLknob  "FREQ1", 20, 10000, -1,1, -1, 50, 0,0
gkfreq2,    iknob2 FLknob  "FREQ2", 20, 10000, -1,1, -1, 50, 50,0
gkfreq3,    iknob3 FLknob  "FREQ3", 20, 10000, -1,1, -1, 50, 100,0
gkfreq4,    iknob4 FLknob  "FREQ4", 20, 10000, -1,1, -1, 50, 150,0

FLsetVal_i   20.0, iknob1
FLsetVal_i   20.0, iknob2
FLsetVal_i   20.0, iknob3
FLsetVal_i   20.0, iknob4

FLpanel_end	;***** end of container

FLrun		;***** runs the widget thread 


gkfreq1 init 20
gkfreq2 init 20
gkfreq3 init 20
gkfreq4 init 20

    instr 1
        aa1 oscili 0.25,gkfreq1,1
        aa2 oscili 0.25,gkfreq2,1
        aa3 oscili 0.25,gkfreq3,1
        aa4 oscili 0.25,gkfreq4,1
        outq aa1,aa2,aa3,aa4
    endin        

