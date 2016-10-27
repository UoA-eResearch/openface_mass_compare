# openface_mass_compare
An openface script that runs a REST server. Posted images are compared against a large dataset, and the most likely match is returned. Works with https://hub.docker.com/r/bamos/openface/

## Setup

Create a folder called "images" in the same directory as this script. Create a folder for each person, and place at least one image for each person in their folder  
Add a file called "data.json" containing additional information about each person in the same folder as this script

## Installation

`sudo su`  
`apt-get install docker`  
`docker pull bamos/openface`  
``docker run --tty=true --interactive=true --volume="`pwd`:/root/my_data" --publish="8000:8000" bamos/openface /bin/bash``  
The volume command mounts the real directory on the left of the colon to the /root/my_data directory in the container  
You should have a terminal within the container. Your PS1 will have changed to something like "root@078226697e7d" to reflect this  
`cd /root/my_data`  
`pip install gunicorn bottle`  
`PYTHONUNBUFFERED=1 gunicorn compare:app --workers 1 --bind=0.0.0.0:8000 --access-logfile - --reload`  
The first time you run this, it'll create a 2D matrix, where each element is an image from the images folder, processed by the Torch7 network  
Once it's done, it'll save the result to data.pickle, for faster startup next time. For a dataset of 3000 images building this pickle file takes ~13 minutes  
If everything looks ok, you can kill this with CTRL-C and add as many workers as you wish (replace --workers 1 with --workers 8 for example)  
To test, run  
`time curl localhost:8000 --data-binary @image.jpg -vv`  
CTRL-P + CTRL-Q to detach from the docker container  
