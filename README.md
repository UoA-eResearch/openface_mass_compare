# openface_mass_compare
An openface script that runs a REST server. Posted images are compared against a large dataset, and the most likely match is returned. Works with https://hub.docker.com/r/uoacer/openface-mass-compare/

## Setup

Create a folder called "images". Create a folder for each person, and place at least one image for each person in their folder  
Add a file called "data.json" containing additional information about each person in the same folder as this script

## Installation

`sudo su`  
`apt-get install docker`  
`docker pull uoacer/openface-mass-compare`  
``docker run --name=omc --restart=always --detach=true --volume="`pwd`:/root/data" --publish="8000:8000" uoacer/openface-mass-compare``  
The volume command mounts the real directory on the left of the colon to the /root/data directory in the container  
The first time you run this, it'll create a 2D matrix, where each element is an image from the images folder, processed by the Torch7 network  
Once it's done, it'll save the result to data.pickle, for faster startup next time. For a dataset of 3000 images building this pickle file takes ~13 minutes  
To test, run  
`time curl localhost:8000 --data-binary @image.jpg -vv`  
