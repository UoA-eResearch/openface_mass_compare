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

from util import *

import numpy as np
np.set_printoptions(precision=2)

from bottle import *
BaseRequest.MEMFILE_MAX = 1e8

with open("/root/data/data.pickle") as f:
    start = time.time()
    reps = pickle.load(f)
    print("Loaded stored pickle, took {}".format(time.time() - start))

data_dict = {}

try:
    with open('/root/data/data.json') as f:
        data = json.load(f)

    if type(data) is dict:
        data_dict = data
    else:
        for d in data['profiles']:
            data_dict[d['upi']] = d
except Exception as e:
    print("Unable to load data.json: " + e)

@get('/')
def default_get():
    return "POST me an image to get the closest match: e.g. time curl localhost:8080 --data-binary @image.jpg -vv\n"

@get('/<uid>')
def get_face(uid):
    f = glob.glob("/root/data/images/{}/*".format(uid))
    return static_file(f[0], '.')

@post('/')
def compare_image():
    response.content_type = 'application/json'

    img_array = np.asarray(bytearray(request.body.read()), dtype=np.uint8)
    print("recieved image of size {}".format(len(img_array)))
    image_data = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image_data is None:
        print("Unable to decode posted image!")
        response.status = 500
        return json.dumps({'error': 'Unable to decode posted image!'})
    try:
        start = time.time()
        rep = getRep(image_data)
        print("Got face representation in {} seconds".format(time.time() - start))
    except:
        response.status = 500
        return json.dumps({'error': 'No face detected'})
    ids_to_compare = request.params.get('ids_to_compare', reps.keys())
    best = 4
    bestUid = "unknown"
    for i in ids_to_compare:
        if type(reps[i]) is not list:
            reps[i] = [reps[i]]
        for r in reps[i]:
            d = rep - r
            dot = np.dot(d,d)
            if dot < best:
                best = dot
                bestUid = i
    return {"uid": bestUid, "confidence": 1 - best/4, "data": data_dict.get(bestUid)}

port = int(os.environ.get('PORT', 8080))

if __name__ == "__main__":
    run(host='0.0.0.0', port=port, debug=True, server='gunicorn', workers=4)

app = default_app()
