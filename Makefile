
runtest1:	kinect-mixer.py
	/usr/bin/fakenect /media/hindle1/Static/hindle1/kinect/video/abram-move-stand-still-move-still-move-raw  python kinect-mixer.py -p

live:	kinect-mixer.py
	python kinect-mixer.py
