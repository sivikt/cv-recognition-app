import os
import logging

logger = logging.getLogger(__name__)

_basedir = os.path.abspath(os.path.dirname(__file__))

DEFAULT_CONFIG = dict(
    HOST='localhost',
    DEBUG=False,
    SERVER_PORT=8180,
    USE_OPENVINO_INFERENCE_ENGINE=False,
    TF_CONFIG={
        'INTER_OP': None,#1,
        'INTRA_OP': None #10
    },
    IMAGE_VALIDATION_CONFIG={
        'MIN_WIDTH': 1024.0,
        'MIN_HEIGHT': 768.0,
        'MIN_WIDTH_DEVIATION': 0.6,
        'MIN_HEIGHT_DEVIATION': 0.6,
        'BLURRINESS_THRESHOLD': 60.0,
        'CAR_SCORE_THRESHOLD': 0.90,
        'CAR_MIN_AREA_THRESHOLD': 0.30,
        'TWO_BIGGEST_CARS_RATIO_THRESHOLD': 2.5,
        'CAR_CENTER_DEVIATION_THRESHOLD': 0.2,
        'NORMALITY_TEST_CHI_SQUARED_PROB_THRESHOLD': 3.27207e-80,
        'DARK_THRESHOLD': 0.3,
        'BRIGHT_THRESHOLD': 0.3
    },
    IMAGE_QUALITY_REQUIREMENTS={
        'requirements': [
            {
                'group': 'The image must be of good quality:',
                'items': [
                    'big enouph (at least 1024x768);',
                    'sharp;',
                    'well illuminated (not to bright or dark);',
                    'contrast.'
                ]
            },
            {
                'group': 'Car on the image must present and:',
                'items': [
                    'takes more than ~40% of the image area;',
                    'centered;',
                    'not be mixed with other cars.'
                ]
            }
        ]
    }
)

#logger.info('Add MKL, KMP and OMP settings')
#os.environ['OMP_NUM_THREADS'] = '44'
#os.environ['KMP_BLOCKTIME'] = '0'
#os.environ['KMP_SETTINGS'] = '1'
#os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
#os.environ['MKL_VERBOSE'] = '1'
#os.environ['MKLDNN_VERBOSE'] = '1'
