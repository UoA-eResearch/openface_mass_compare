#!/usr/bin/env python2

from util import *

def loadImageFromFile(imgPath):
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    return getRep(bgrImg)

reps = {}

g = glob.glob("/root/data/images/*/*")

start = time.time()

for f in g:
    uid = os.path.splitext(os.path.basename(f))[0]
    try:
        reps[uid] = loadImageFromFile(f)
    except:
        pass

print("Loaded {}/{} refs, took {} seconds.".format(len(reps), len(g), time.time() - start))

with open("/root/data/data.pickle", 'wb') as f:
    pickle.dump(reps, f) 
