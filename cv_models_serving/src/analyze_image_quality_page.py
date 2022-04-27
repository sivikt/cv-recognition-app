import logging

from config.cfg import DEFAULT_CONFIG

from image_validation import validation_consts as vc

if not DEFAULT_CONFIG['USE_OPENVINO_INFERENCE_ENGINE']:
    from image_validation import image_validation as imgval

    img_validator = imgval.ValidationBase(
        inter_op=DEFAULT_CONFIG['TF_CONFIG']['INTER_OP'],
        intra_op=DEFAULT_CONFIG['TF_CONFIG']['INTRA_OP']
    )
else:
    from image_validation import image_validation_openvino as imgval

    img_validator = imgval.ValidationOpenVino()


from util.timer import create_elapsed_timer


import numpy as np
import traceback
import base64
import cv2

from flask import current_app as app
from flask import Blueprint
from flask import request
from flask import json
from flask import Response


logger = logging.getLogger(__name__)


mod = Blueprint('analyze_image_quality_page', __name__, url_prefix='/img/analyse')


IMAGE_VALIDATION_ERROR_CODES = {
    vc.LOW_IMAGE_QUALITY_SMALL_ERROR: 'LOW_IMAGE_QUALITY_SMALL_ERROR',
    vc.LOW_IMAGE_QUALITY_BLURRY_ERROR: 'LOW_IMAGE_QUALITY_BLURRY_ERROR',
    vc.LOW_IMAGE_QUALITY_DARK_ERROR: 'LOW_IMAGE_QUALITY_DARK_ERROR',
    vc.LOW_IMAGE_QUALITY_BRIGHT_ERROR: 'LOW_IMAGE_QUALITY_BRIGHT_ERROR',
    vc.LOW_IMAGE_QUALITY_BAD_ILLUMINATION_ERROR: 'LOW_IMAGE_QUALITY_BAD_ILLUMINATION_ERROR',
    vc.LOW_IMAGE_QUALITY_NOT_CONTRAST_ERROR: 'LOW_IMAGE_QUALITY_NOT_CONTRAST_ERROR',
    vc.LOW_CAR_QUALITY_NO_CAR_ERROR: 'LOW_CAR_QUALITY_NO_CAR_ERROR',
    vc.LOW_CAR_QUALITY_SMALL_CAR_ERROR: 'LOW_CAR_QUALITY_SMALL_CAR_ERROR',
    vc.LOW_CAR_QUALITY_NOT_CENTERED_CAR_ERROR: 'LOW_CAR_QUALITY_NOT_CENTERED_CAR_ERROR',
    vc.LOW_CAR_QUALITY_CAR_AMBIGUITY_ERROR: 'LOW_CAR_QUALITY_CAR_AMBIGUITY_ERROR'
}


IMAGE_VALIDATION_ERROR_MESSAGES = {
    vc.LOW_IMAGE_QUALITY_SMALL_ERROR: 'The photo is of low quality.\nIt\'s too small',
    vc.LOW_IMAGE_QUALITY_BLURRY_ERROR: 'The photo is of low quality.\nIt\'s too blurry',
    vc.LOW_IMAGE_QUALITY_DARK_ERROR: 'The photo is of low quality.\nIt\'s too dark',
    vc.LOW_IMAGE_QUALITY_BRIGHT_ERROR: 'The photo is of low quality.\nIt\'s too bright',
    vc.LOW_IMAGE_QUALITY_BAD_ILLUMINATION_ERROR: 'The photo is of low quality.\nIt\'s badly illuminated',
    vc.LOW_IMAGE_QUALITY_NOT_CONTRAST_ERROR: 'The photo is of low quality.\nIt\'s not contrast',
    vc.LOW_CAR_QUALITY_NO_CAR_ERROR: 'Car on the photo is absent',
    vc.LOW_CAR_QUALITY_SMALL_CAR_ERROR: 'Car on the photo is too small',
    vc.LOW_CAR_QUALITY_NOT_CENTERED_CAR_ERROR: 'Car on the photo must be centered',
    vc.LOW_CAR_QUALITY_CAR_AMBIGUITY_ERROR: 'Too many cars on the photo'
}


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                            np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_image_validation_error_code(error_id):
    if error_id == vc.GOOD_IMAGE:
        return None
    else:
        return IMAGE_VALIDATION_ERROR_CODES[error_id]


def get_image_validation_error_message(error_id):
    if error_id == vc.GOOD_IMAGE:
        return None
    else:
        return IMAGE_VALIDATION_ERROR_MESSAGES[error_id]


def create_api_error(error_code='unknown', error_desc='unknown'):
    return {
        'error_code': error_code,
        'error_desc': error_desc
    }


def calc_IoU(bbox1, bbox2):
    ix_min = max(bbox1[1], bbox2[1])
    iy_min = max(bbox1[0], bbox2[0])
    ix_max = min(bbox1[3], bbox2[3])
    iy_max = min(bbox1[2], bbox2[2])

    i_area = max(0, ix_max - ix_min) * max(0, iy_max - iy_min)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    return i_area / (bbox1_area + bbox2_area - i_area)


@mod.route('/image_quality_requirements', methods=['GET'])
def image_quality_requirements():
    return Response(json.dumps(app.config['IMAGE_QUALITY_REQUIREMENTS']), status=200, mimetype='application/json')


@mod.route('/make_predictions', methods=['GET'])
def make_predictions():
    total_sw = create_elapsed_timer('sec')

    is_developer_mode = request.args.get('is_developer_mode')
    is_developer_mode = True if is_developer_mode is not None and is_developer_mode.lower() == 'true' else False

    skip_validation = request.args.get('skip_validation')
    skip_validation = True if skip_validation is not None and skip_validation.lower() == 'true' else False

    include_orig_img = request.args.get('include_orig_img')
    include_orig_img = True if include_orig_img is not None and include_orig_img.lower() == 'true' else False

    src_rgb_img_path = request.args.get('image_path')
    
    logger.debug('IN %s got arguments skip_validation=%s, include_orig_img=%s, src_rgb_img_path=%s',
                 make_predictions.__name__, skip_validation, include_orig_img, src_rgb_img_path)

    try:
        image_np = cv2.imread(src_rgb_img_path)

        err_code = None
        err_msg = None

        if not skip_validation:
            err_id, val_results = img_validator.validate_image(image=image_np, image_path=src_rgb_img_path,
                                                               validation_cfg=app.config['IMAGE_VALIDATION_CONFIG'])
            err_code = get_image_validation_error_code(err_id)
            err_msg = get_image_validation_error_message(err_id)

        if skip_validation or (not err_code):
            predictions = {
                'data': {
                    'width': image_np.shape[0],
                    'height': image_np.shape[1],
                    'bboxes': [],
                    'scores': [],
                    'classes': []
                }
            }

            return Response(json.dumps(predictions, cls=NumpyEncoder), status=200, mimetype='application/json')
        elif not skip_validation:
            err = create_api_error(err_code, err_msg)
            logger.exception(err)

            if include_orig_img:
                with open(src_rgb_img_path, "rb") as img:
                    img_base64 = base64.b64encode(img.read())
                    err['img_src'] = img_base64.decode("utf-8")

            return Response(json.dumps(err),
                            status=400,
                            mimetype='application/json')
    except Exception:
        err_msg = traceback.format_exc()
        logger.exception(err_msg)
        return Response(json.dumps(create_api_error('ise', err_msg)), status=500,
                        mimetype='application/json')
    finally:
        logger.debug('IN %s done in %s', make_predictions.__name__, total_sw())
