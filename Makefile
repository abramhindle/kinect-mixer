
runtest1:	kinect-mixer.py
	/usr/bin/fakenect /media/hindle1/Static/hindle1/kinect/video/abram-move-stand-still-move-still-move-raw  python kinect-mixer.py -p

live:	kinect-mixer.py
	python kinect-mixer.py

oscrelay: oscrelay.pl
	hypnotoad -f oscrelay.pl

mixerpanel:
	setsid firefox http://127.0.0.1:8080/connectivity.html &

testsig: testsig.sco testsig.orc
	csound  -+rtaudio=jack -o devaudio -b 400 -B 2048 -L stdin testsig.orc testsig.sco

