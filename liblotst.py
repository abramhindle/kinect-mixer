import liblo
import argparse
import random
parser = argparse.ArgumentParser(description='Lo!')
parser.add_argument('-osc', dest='osc', default=7770,help="OSC Port")
parser.set_defaults(osc=7770)
args = parser.parse_args()

target = liblo.Address(args.osc)
def send_osc(path,*args):
    global target
    return liblo.send(target, path, *args)


send_osc("/kinect/centroid",random.uniform(0,3.0),random.uniform(1.0,2.0),3.0)
