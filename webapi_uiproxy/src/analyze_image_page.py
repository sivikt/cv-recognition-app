from util.timer import create_elapsed_timer

import numpy as np
import traceback
import uuid
from pathlib import Path

from flask import current_app as app
from flask import Blueprint
from flask import request
from flask import json
from flask import Response
from werkzeug.utils import secure_filename
from PIL import Image

import requests

mod = Blueprint('analyze_image_page', __name__, url_prefix='/image/analyse')


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                            np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def allowed_file(filename):
    with Image.open(filename) as img:
        format = img.format
        format = format.lower() if format is not None else None
    print('source image', filename, ' is in ', format, ' format')
    return format in app.config.get('ALLOWED_IMAGE_FORMATS')


def create_api_error(error_code='unknown', error_desc='unknown'):
    return {
        'error_code': error_code,
        'error_desc': error_desc
    }


def models_endpoint(path):
    return app.config['MODELS_API_ENDPOINT']+path


@mod.route('/objects_detection_classes_index', methods=['GET'])
def objects_detection_classes_index():
    resp = requests.get(url=models_endpoint('/objects_detection_classes_index'), timeout=5*60)
    return Response(resp.text, status=resp.status_code, mimetype='application/json')


@mod.route('/image_quality_requirements', methods=['GET'])
def image_quality_requirements():
    resp = requests.get(url=models_endpoint('/image_quality_requirements'), timeout=5*60)
    return Response(
        resp.text,
        status=resp.status_code,
        mimetype='application/json'
    )


@mod.route('/gallery', methods=['GET'])
def get_gallery_images():
    return Response(
        json.dumps(
            {'data': list(app.config['DEMO_DATASET_INDEX'].keys())}
        ),
        status=200,
        mimetype='application/json'
    )


@mod.route('/find_objects', methods=['POST'])
def find_objects():
    total_sw = create_elapsed_timer('sec')

    try:
        img_prepare_sw = create_elapsed_timer('sec')

        if 'file' not in request.files:
            return Response(
                json.dumps(
                    create_api_error('image_file_required', 'No file part')
                ),
                status=400,
                mimetype='application/json'
            )

        file = request.files['file']

        if not file.filename.strip():
            return Response(
                json.dumps(
                    create_api_error('image_file_required', 'No selected file')
                ),
                status=400,
                mimetype='application/json'
            )

        filename = Path(secure_filename(file.filename))

        unique_prefix = str(uuid.uuid4())
        orig_file_path = str(Path(app.config['IMAGES_UPLOAD_FOLDER']) / (unique_prefix + '_orig_' + filename.name))
        file.save(orig_file_path)
        print(__name__, 'GOT', file.filename, 'SAVE TO', orig_file_path)

        if not allowed_file(orig_file_path):
            print(__name__, 'INVALID', orig_file_path)
            return Response(
                json.dumps(
                    create_api_error(
                        'invalid_file_format',
                        f'Only next image formats are allowed {app.config.get("ALLOWED_IMAGE_FORMATS")}'
                    )
                ),
                status=400,
                mimetype='application/json'
            )

        with Image.open(orig_file_path) as img:
            src_rgb_img_path = str(Path(app.config['IMAGES_UPLOAD_FOLDER']) / (unique_prefix + filename.stem + '.jpg'))
            img = img.convert('RGB')
            img.save(src_rgb_img_path, quality=100)

            print(__name__, 'CONVERTED ORIGINAL IMAGE', orig_file_path, 'TO RGB', src_rgb_img_path)

        img_prepare_sw('image preparation (save, convert)')

        is_developer_mode = request.args.get('is_developer_mode')
        is_developer_mode = True if is_developer_mode is not None and is_developer_mode.lower() == 'true' else False

        skip_validation = request.args.get('skip_validation')
        skip_validation = True if skip_validation is not None and skip_validation.lower() == 'true' else False

        not_include_orig_img = request.args.get('not_include_orig_img')
        include_orig_img = False if not_include_orig_img is not None and not_include_orig_img.lower() == 'true' else True

        print('IN got arguments skip_validation=', skip_validation, ' include_orig_img=', include_orig_img)

        resp = requests.get(
            url=models_endpoint(f'/make_predictions?image_path={src_rgb_img_path}&is_developer_mode={is_developer_mode}&skip_validation={skip_validation}&include_orig_img={include_orig_img}'),
            timeout=5*60
        )
        return Response(resp.text, status=resp.status_code, mimetype='application/json')
    except Exception:
        err_msg = traceback.format_exc()
        print(err_msg)
        return Response(json.dumps(create_api_error('ise', err_msg)), status=500,
                        mimetype='application/json')
    finally:
        total_sw(find_objects.__name__)
