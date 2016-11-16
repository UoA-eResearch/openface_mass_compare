#!/bin/bash

cd /root

if [ ! -f "/root/data/data.pickle" ];then
  echo generating pickle
  python generate_pickle.py
fi

PYTHONUNBUFFERED=1 gunicorn web_server:app --workers 8 --bind=0.0.0.0:8000 --access-logfile - --reload