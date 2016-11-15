FROM bamos/openface
RUN pip install gunicorn bottle
COPY compare.py /root/compare.py
CMD cd /root; PYTHONUNBUFFERED=1 gunicorn compare:app --workers 8 --bind=0.0.0.0:8000 --access-logfile - --reload
