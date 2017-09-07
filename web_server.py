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

from bottle import *
BaseRequest.MEMFILE_MAX = 1e8

@get('/')
def default_get():
    return static_file("index.html", ".")

@get('/<uid>')
def get_face(uid):
    f = glob.glob("/root/data/images/{}/*".format(uid))
    return static_file(f[0], '/')

@post('/')
def compare_image():
    if request.files.get('pic'):
        binary_data = request.files.get('pic').file.read()
    else:
        binary_data = request.body.read()

    img_array = np.asarray(bytearray(binary_data), dtype=np.uint8)
    print("recieved image of size {}".format(len(img_array)))
    image_data = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image_data is None:
        print("Unable to decode posted image!")
        response.status = 500
        return {'error': 'Unable to decode posted image!'}
    try:
        start = time.time()
        result = getPeople(image_data)
        print("Got face representation in {} seconds".format(time.time() - start))
        return json.dumps(result, indent=4)
    except Exception as e:
        print("Error: {}".format(e))
        response.status = 500
        return {'error': str(e)}

port = int(os.environ.get('PORT', 8080))

if __name__ == "__main__":
    run(host='0.0.0.0', port=port, debug=True, server='gunicorn', workers=4)

app = default_app()
