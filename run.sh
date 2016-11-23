#!/bin/bash

cd /root

export PYTHONUNBUFFERED=1

if [ ! -f "/root/data/data.pickle" ];then
  echo generating pickle
  python generate_pickle.py
fi

gunicorn web_server:app --workers 8 --bind=0.0.0.0:8000 --access-logfile - --reload