#!/bin/sh

CV_MODELS_SERVING_HOME=/home/cv-recognition-app/cv_models_serving
WEBAPI_UIPROXY_HOME=/home/cv-recognition-app/webapi_uiproxy

sudo apt-get install nginx
sudo apt-get install python3-pip
sudo pip3 install virtualenv

cd $CV_MODELS_SERVING_HOME
virtualenv -p python3 .env
source .env/bin/activate
pip3 install -r requirements.txt
source .env/bin/deactivate

cd $WEBAPI_UIPROXY_HOME
virtualenv -p python3 .env
source .env/bin/activate
pip3 install -r requirements.txt

##
# NGINX deployment
##
sudo nginx -t
sudo service nginx restart