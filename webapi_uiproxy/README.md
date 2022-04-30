#### REQUIREMENTS:
- python 3
- pip 3

#### SETUP ENV:
```sh
virtualenv -p python3 .env
source .env/bin/activate
pip3 install -r requirements.txt
```

#### RUN:
```sh
cd src
source .env/bin/activate
python3 .
```
or
```sh
gunicorn webclient_app:app -b localhost:8080 --workers 2 --timeout 3600 --access-logfile gunicorn-access.log --error-logfile gunicorn-error.log
```
