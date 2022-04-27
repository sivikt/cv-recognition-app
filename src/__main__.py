import argparse

import cv_recognition_app

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--host', required=False, default='localhost', help='server hostname')
parser.add_argument('--debug', required=False, default=False, help='is debug mode')
#parser.add_argument('--port', required=False, default='8080', help='server port')

args = parser.parse_args()
args_config = {
   'HOST':  args.host,
   #'SERVER_PORT':  args.port,
   'DEBUG':  args.debug
}


cv_recognition_app.start(args_config)
