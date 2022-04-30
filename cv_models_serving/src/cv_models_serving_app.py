import logging
from logging.handlers import RotatingFileHandler
import sys

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%d-%b-%Y %H:%M:%S:%z',
    handlers=[
        RotatingFileHandler('intel_test.log', maxBytes=(20 * 1 * 1024 * 1 * 1024), backupCount=10),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.getLogger('PIL.Image').setLevel(logging.WARNING)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


from flask import Flask

from config.cfg import DEFAULT_CONFIG

app = Flask(__name__)
app.config.from_object(__name__)
app.config.update(DEFAULT_CONFIG)
app.config.from_envvar('FLASK_SETTINGS', silent=True)

logger.info('Application config %s', app.config)


from analyze_image_quality_page import mod as analyze_image_quality_page


app.register_blueprint(analyze_image_quality_page)


def start(args_config):
    logger.info('Starting application')
    app.config.update(args_config)
    app.run(host=app.config.get('HOST'), port=app.config.get('SERVER_PORT'), debug=app.config.get('DEBUG'))
