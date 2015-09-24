
sr = 44100
kr = 441
ksmps = 100
nchnls = 4



FLcolor	180,200,199
FLpanel 	"Mixer",200,100
            
gkamp1,    iknob1 FLknob  "AMP1", 0.0001, 2, -1,1, -1, 50, 0,0
gkamp2,    iknob2 FLknob  "AMP2", 0.0001, 2, -1,1, -1, 50, 50,0
gkamp3,    iknob3 FLknob  "AMP3", 0.0001, 2, -1,1, -1, 50, 100,0
gkamp4,    iknob4 FLknob  "AMP4", 0.0001, 2, -1,1, -1, 50, 150,0

FLsetVal_i   1.0, iknob1
FLsetVal_i   1.0, iknob2
FLsetVal_i   1.0, iknob3
FLsetVal_i   1.0, iknob4

FLpanel_end	;***** end of container

FLrun		;***** runs the widget thread 


gkamp1 init 0.0001
gkamp2 init 0.0001
gkamp3 init 0.0001
gkamp4 init 0.0001

    instr 1
        a1,a2,a3,a4 inq
        aa1 = a1 * gkamp1
        aa2 = a2 * gkamp2
        aa3 = a3 * gkamp3
        aa4 = a4 * gkamp4
        outq aa1,aa2,aa3,aa4
    endin        

gihandle OSCinit 7770

    instr oscmix       
        kf1 init 0
        kf2 init 0
        kf3 init 0
        kf4 init 0
      nxtmsg:           
        kk  OSClisten gihandle, "/foo/bar", "ffff", kf1, kf2, kf3, kf4
      if (kk == 0) goto ex
        gkamp1 = kf1  
        gkamp2 = kf2  
        gkamp3 = kf3  
        gkamp4 = kf4
        kgoto nxtmsg
      ex:
    endin
