import pathlib

from flask import Flask, redirect, url_for

from config.cfg import DEFAULT_CONFIG

app = Flask(__name__, static_folder='static', template_folder="views")
app.config.from_object(__name__)
app.config.update(DEFAULT_CONFIG)
app.config.from_envvar('FLASKR_SETTINGS', silent=True)

IMAGES_UPLOAD_FOLDER_PATH = pathlib.Path(app.config.get('IMAGES_UPLOAD_FOLDER'))
IMAGES_UPLOAD_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

DEMO_DATASET_FOLDER_PATH = pathlib.Path(app.config.get('DEMO_DATASET_FOLDER'))
DEMO_DATASET = list(DEMO_DATASET_FOLDER_PATH.glob('**/*.jpg'))

app.config['DEMO_DATASET'] = DEMO_DATASET
app.config['DEMO_DATASET_INDEX'] = {str(pathlib.Path(p.parent.name)/p.name): p for p in DEMO_DATASET}


from pages import mod as pages_mod
from analyze_car_damage_page import mod as analyze_car_damage_page_mod


app.register_blueprint(pages_mod)
app.register_blueprint(analyze_car_damage_page_mod)


@app.route('/', methods=['GET'])
def __index():
    return redirect(url_for('pages.main'))


def start(args_config):
    print('Starting application')
    app.config.update(args_config)
    app.run(host=app.config.get('HOST'), port=app.config.get('SERVER_PORT'), debug=app.config.get('DEBUG'))
