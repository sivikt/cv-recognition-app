#!/bin/sh

sudo apt-get install nginx
sudo apt-get install python3-pip
sudo pip3 install virtualenv

virtualenv -p python3 .env
source .env/bin/activate

pip3 install -r requirements.txt


##
# NGinx deployment
##
sudo cp car_dmg_estimation-demo-webapp /etc/nginx/sites-available/default
sudo nginx -t
sudo service nginx restart