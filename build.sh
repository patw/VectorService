#!/bin/bash

echo
echo "+================================"
echo "| START: Vector Service"
echo "+================================"
echo

echo 
echo "Building container"
echo
docker build -t graboskyc/vectorservice:latest .

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
