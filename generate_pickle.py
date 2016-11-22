#!/usr/bin/env python2

from util import *
from multiprocessing import Pool, cpu_count
from threading import local

def init():
    global align, net
    align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), 96)

def loadImageFromFile(imgPath):
    global align, net
    uid = os.path.split(os.path.split(s)[0])[-1]
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        print("Unable to load image: {}".format(imgPath))
        return
    try:
        rep = getRep(bgrImg, align, net)
    except Exception as e:
        print('{} for {}'.format(e, uid))
        return
    return (uid, rep)

PROCESSES = cpu_count() / 2
p = Pool(processes=PROCESSES, initializer=init)
g = glob.glob("/root/data/images/*/*")

start = time.time()
reps = p.imap_unordered(loadImageFromFile, g)

rep_dict = {}

count = 0
successes = 0
for r in reps:
    count += 1
    if count % 100 == 0:
        print("{}s: {}/{} done".format(time.time() - start, count, len(g)))
    if r:
        successes += 1
        if r[0] in rep_dict:
            rep_dict[r[0]].append(r[1])
        else:
            rep_dict[r[0]] = [r[1]]

print("Loaded {}/{} refs, took {} seconds.".format(successes, len(g), time.time() - start))

with open("/root/data/data.pickle", 'wb') as f:
    pickle.dump(rep_dict, f)
