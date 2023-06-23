#!/bin/bash

echo
echo "+================================"
echo "| START: Vector Service"
echo "+================================"
echo

datehash=`date | md5sum | cut -d" " -f1`
abbrvhash=${datehash: -8}

echo 
echo "Building container using tag ${abbrvhash}"
echo
docker build -t graboskyc/vectorservice:latest -t graboskyc/vectorservice:${abbrvhash} .

echo 
echo "Starting container"
echo
docker stop vectorservice
docker rm vectorservice
docker run -t -i -d -p 80:80 --name vectorservice --restart unless-stopped graboskyc/vectorservice:latest

echo
echo "+================================"
echo "| END:  Vector Service"
echo "+================================"
echo
