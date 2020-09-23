#!/bin/bash

docker network create oc-network

docker run --rm -it -p8000:8000 -p8001:8001 -p8002:8002 \ # add --gpus=all to turn on GPUs, you'll need nvidia driver 450.51
       -v`pwd`/oc_models:/oc_models \
       --network oc-network \
       --name ocserver \
       local/tritonserver:20.08-tfpepr-py3 \
       tritonserver --model-repository=/oc_models --backend-config=tensorflow,version=2 --log-verbose=1 --log-error=1 --log-info=1
