#!/usr/bin/env python
import cv2
import os
import numpy as np

import openface

import glob
import time
import pickle
import json
import sys

modelDir = os.path.join('/root/openface', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), 96)

data_dict = {}

try:
    with open("/root/data/data.pickle") as f:
        start = time.time()
        reps = pickle.load(f)
        print("Loaded stored pickle, took {}".format(time.time() - start))
except Exception as e:
    print("Unable to load data.pickle: ", e)

try:
    with open('/root/data/data.json') as f:
        data = json.load(f)

    if 'profiles' in data:
        for d in data['profiles']:
            if 'upi' in d:
                data_dict[d['upi']] = d
    else:
        data_dict = data
except Exception as e:
    print("Unable to load data.json: ", e)

def getRep(bgrImg, align=align, net=net):
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face")
    alignedFace = align.align(96, rgbImg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image")
    rep = net.forward(alignedFace)
    return rep

def getPeople(bgrImg, align=align, net=net):
    faces = []
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bb = align.getAllFaceBoundingBoxes(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face")
    for face in bb:
        alignedFace = align.align(96, rgbImg, face, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            print("Unable to align image")
            continue
        if not alignedFace is None:
            rep = net.forward(alignedFace)
            best = 4
            bestUid = "unknown"
            for i in reps.keys():
                if type(reps[i]) is not list:
                    reps[i] = [reps[i]]
                for r in reps[i]:
                    d = rep - r
                    dot = np.dot(d,d)
                    if dot < best:
                        best = dot
                        bestUid = i
            faces.append({
              "face_rectangle": {
                "left": face.left(),
                "top": face.top(),
                "width": face.width(),
                "height": face.height()
              },
              "uid": bestUid,
              "confidence": 1 - best/4,
              "data": data_dict.get(bestUid)
            })
    return faces

if __name__ == "__main__":
  image = cv2.imread(sys.argv[1])
  import pprint
  pprint.pprint(getPeople(image))
