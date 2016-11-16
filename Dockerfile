FROM bamos/openface
MAINTAINER Nick Young <nick.young@auckland.ac.nz>
RUN pip install gunicorn bottle
COPY * /root/
CMD /root/run.sh
