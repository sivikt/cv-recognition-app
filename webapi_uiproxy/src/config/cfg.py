import os
import pathlib


_basedir = os.path.abspath(os.path.dirname(__file__))

DEFAULT_CONFIG = dict(
    HOST='localhost',
    DEBUG=False,
    SERVER_PORT=8080,
    IMAGES_UPLOAD_FOLDER=str(pathlib.Path(__file__).parent.parent / 'uploaded_images'),
    DEMO_DATASET_FOLDER=str(pathlib.Path(__file__).parent.parent / 'static' / 'demo_dataset'),
    ALLOWED_IMAGE_FORMATS={'jpeg', 'png', 'bmp'},
    MODELS_API_ENDPOINT='http://localhost:8180'
)

del os