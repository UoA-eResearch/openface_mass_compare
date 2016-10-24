#!/usr/bin/env python2
#
# Adapted from: Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To test:
# time curl localhost:8080 --data-binary @image.jpg -vv

import time
import argparse
import cv2
import itertools
import os

import numpy as np
np.set_printoptions(precision=2)

import openface

from bottle import *
BaseRequest.MEMFILE_MAX = 1e8
import glob
import pickle
import json

modelDir = os.path.join('/root/openface', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), 96)

picklefile = "data.pickle"

def loadImageFromFile(imgPath):
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    return getRep(bgrImg)

def getRep(bgrImg):
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face")
    alignedFace = align.align(96, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image")
    rep = net.forward(alignedFace)
    return rep

if os.path.isfile(picklefile):
    with open(picklefile) as f:
        start = time.time()
        reps = pickle.load(f)
        print("Loaded stored pickle, took {}".format(time.time() - start))

else:
    reps = {}

    g = glob.glob("images/*/*")

    start = time.time()

    for f in g:
        uid = os.path.splitext(os.path.basename(f))[0]
        try:
            reps[uid] = loadImageFromFile(f)
        except:
            pass

    print("Loaded {}/{} refs, took {} seconds.".format(len(reps), len(g), time.time() - start))

    with open(picklefile, 'wb') as f:
        pickle.dump(reps, f)

with open('data.json') as f:
  data = json.load(f)

data_dict = {}

for d in data['profiles']:
  data_dict[d['upi']] = d

@get('/')
def default_get():
  return "POST me an image to get the closest match: e.g. time curl localhost:8080 --data-binary @image.jpg -vv\n"

@get('/<uid>')
def get_face(uid):
  f = glob.glob("images/{}/*".format(uid))
  return static_file(f[0], '.')

@post('/')
def compare_image():
  img_array = np.asarray(bytearray(request.body.read()), dtype=np.uint8)
  print("recieved image of size {}".format(len(img_array)))
  image_data = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
  if image_data is None:
    print("Unable to decode posted image!")
    abort(500, "Unable to decode posted image")
  try:
    start = time.time()
    rep = getRep(image_data)
    print("Got face representation in {} seconds".format(time.time() - start))
  except:
    abort(500, "No face detected")
  ids_to_compare = request.params.get('ids_to_compare', reps.keys())
  best = 4
  bestUid = "unknown"
  for i in ids_to_compare:
      d = rep - reps[i]
      dot = np.dot(d,d)
      if dot < best:
          best = dot
          bestUid = i
  return {"uid": bestUid, "confidence": 1 - best/4, "data": data_dict[bestUid]}

port = int(os.environ.get('PORT', 8080))

if __name__ == "__main__":
  run(host='0.0.0.0', port=port, debug=True, server='gunicorn', workers=4)

app = default_app()