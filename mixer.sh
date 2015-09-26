( sleep 6; \
        jack_connect csound6:output1 system:playback_1 ; \
        jack_connect csound6:output2 system:playback_1 ; \
        jack_connect csound6:output3 system:playback_1 ; \
        jack_connect csound6:output4 system:playback_1 ; \
        jack_connect csound6:output1 system:playback_2 ; \
        jack_connect csound6:output2 system:playback_2 ; \
        jack_connect csound6:output3 system:playback_2 ; \
        jack_connect csound6:output4 system:playback_2 ; \
) &

csound -dm6 -+rtaudio=jack -i devaudio -o devaudio -b 400 -B 2048 -L stdin mixer.orc mixer.sco

