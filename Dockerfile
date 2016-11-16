FROM bamos/openface
RUN pip install gunicorn bottle
COPY * /root/
CMD /root/run.sh
