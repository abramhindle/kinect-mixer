
sr = 44100
kr = 441
ksmps = 100
nchnls = 4



FLcolor	180,200,199
FLpanel 	"Mixer",250,100
            
gkamp1,    iknob1 FLknob  "AMP1", 0.0001, 2, -1,1, -1, 50, 0,0
gkamp2,    iknob2 FLknob  "AMP2", 0.0001, 2, -1,1, -1, 50, 50,0
gkamp3,    iknob3 FLknob  "AMP3", 0.0001, 2, -1,1, -1, 50, 100,0
gkamp4,    iknob4 FLknob  "AMP4", 0.0001, 2, -1,1, -1, 50, 150,0
gkexpscale, iknob5 FLknob  "Subtle", 1.00, 4.0, -1,1, -1, 50, 200,0
gkhtim,    islider5 FLslider  "HTime", 0.01, 1.0, -1,1,1, 150,25,0,75
gkignore,  ibutton  FLbutton  "Motion",0,1,22,50,25,150,75,-1

gkignore init 0
FLsetVal_i   0.0, ibutton
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

gkhtim init 0.25


; (-2.5654042855,0.731932929987,1.92851628927)
; INFO:root:Centroid sending (2.89381892107,-0.526489799227,3.85365332503)
; INFO:root:Centroid sending (3.15571526385,-0.879275256892,3.63295835472)
;gkpos1x init -2.5
;gkpos1y init 0.7
;gkpos1z init 1.9
; INFO:root:Centroid sending (2.36762915481,-0.675700425355,3.33034554415)
; INFO:root:Centroid sending (2.89381892107,-0.526489799227,3.85365332503)
; INFO:root:Centroid sending (3.15571526385,-0.879275256892,3.63295835472)

gkpos1x init 2.8
gkpos1y init -0.675
gkpos1z init 3.6

;INFO:root:Centroid sending (-2.00309403727,-0.787678318548,3.20481173845)
; 0.574901953346,-0.14967868418,4.63504335212
; INFO:root:Centroid sending (-2.34979532587,-0.903490026386,3.72836603454)
gkpos2x init -2.349
gkpos2y init -0.9
gkpos2z init 3.7

; WARNING:root:Centroid: (2.51921113936,-0.775117179692,3.31196510232) reached!
; 0.574901953346,-0.14967868418,4.63504335212
gkpos3x init 0.574
gkpos3y init -0.149
gkpos3z init 4.63


giexpscale init 1.5
gisqrt init sqrt(5.0)

    instr 1
        a1,a2,a3,a4 inq
        kamp1 portk gkamp1, gkhtim
        kamp2 portk gkamp2, gkhtim
        kamp3 portk gkamp3, gkhtim
        kamp4 portk gkamp4, gkhtim
        aa1 = a1 * kamp1
        aa2 = a2 * kamp2
        aa3 = a3 * kamp3
        aa4 = a4 * kamp4
        outq aa1,aa2,aa3,aa4
    endin        

gihandle OSCinit 7770

    instr oscmix       
        kf1 init 0
        kf2 init 0
        kf3 init 0
        kf4 init 0
      nxtmsg:           
        kk  OSClisten gihandle, "/mixer/fourch", "ffff", kf1, kf2, kf3, kf4
      if (kk == 0) goto ex
        gkamp1 = kf1  
        gkamp2 = kf2  
        gkamp3 = kf3  
        gkamp4 = kf4
        kgoto nxtmsg
      ex:
    endin

    instr oscequalmix       
        kf1 init 0
        kf2 init 0
        kf3 init 0
        kf4 init 0
      nxtmsg:           
        kk  OSClisten gihandle, "/mixer/equalmix", "f", kf1
      if (kk == 0) goto ex
        gkamp1 = kf1  
        gkamp2 = kf1
        gkamp3 = kf1  
        gkamp4 = kf1
        kgoto nxtmsg
      ex:
    endin


    instr oscposition       
        kf1 init 0
        kf2 init 0
        kf3 init 0
        kf4 init 0
      nxtmsg:           
        kk  OSClisten gihandle, "/kinect/position", "ifff", kf1, kf2, kf3, kf4
      if (kk == 0) goto ex
        if (kf1 == 0) goto pos0
        if (kf1 == 1) goto pos1
        if (kf1 == 2) goto pos2
        kgoto nxtmsg
      pos0:
        gkpos1x = kf2
        gkpos1y = kf3
        gkpos1z = kf4
        kgoto nxtmsg
      pos1:
        gkpos2x = kf2
        gkpos2y = kf3
        gkpos2z = kf4
        kgoto nxtmsg
      pos2:
        gkpos3x = kf2
        gkpos3y = kf3
        gkpos3z = kf4
        kgoto nxtmsg
      ex:
    endin

    instr osccentroid    
        kf1 init 0
        kf2 init 0
        kf3 init 0
        kf4 init 0
        kdist1 init 0
        kdist2 init 0
        kdist3 init 0
      if (gkignore == 1) goto ex
      nxtmsg:           
        kk  OSClisten gihandle, "/kinect/centroid", "fff", kf1, kf2, kf3
      if (kk == 0) goto ex
        kdist1 = sqrt((kf1 - gkpos1x)^2 + (kf2 - gkpos1y)^2 + (kf3 - gkpos1z)^2)
        kdist2 = sqrt((kf1 - gkpos2x)^2 + (kf2 - gkpos2y)^2 + (kf3 - gkpos2z)^2)
        kdist3 = sqrt((kf1 - gkpos3x)^2 + (kf2 - gkpos3y)^2 + (kf3 - gkpos3z)^2)
        gkamp1 = exp(1.0 + (gkexpscale - 1) * (1.0 - (kdist1/gisqrt)))/exp(gkexpscale)
        gkamp2 = exp(1.0 + (gkexpscale - 1) * (1.0 - (kdist2/gisqrt)))/exp(gkexpscale)
        gkamp3 = exp(1.0 + (gkexpscale - 1) * (1.0 - (kdist3/gisqrt)))/exp(gkexpscale)
        printk2 kdist1
        printk2 kdist2
        kgoto nxtmsg
      ex:
    endin
