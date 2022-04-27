import numpy as np
import cv2
import logging

from scipy import stats

from util.timer import create_elapsed_timer
from . import validation_consts as vc
from .car_validation import CarValidatorBase


logger = logging.getLogger(__name__)


class ValidationBase:
    def __init__(self, intra_op=None, inter_op=None):
        self.cars_validator = CarValidatorBase(intra_op=intra_op, inter_op=inter_op)

    def get_validators(self):
        # validators = [check_size, check_car_is_on_image]
        return [
            self.check_size,
            self.check_blurriness,
            self.check_bright_or_dark,
            self.check_histogram_is_normal,
            self.cars_validator.check_car_is_on_image,
        ]

    def __variance_of_laplacian(self, image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def check_size(self, image_path=None, image=None, validation_cfg=vc.DEFAULT_VALIDATION_CONFIG):
        if image is None:
            image = cv2.imread(image_path)

        height, width, depth = image.shape

        result_val = {'height': height, 'width': width}

        if (width/validation_cfg['MIN_WIDTH'] < validation_cfg['MIN_WIDTH_DEVIATION']) or\
           (height/validation_cfg['MIN_HEIGHT'] < validation_cfg['MIN_HEIGHT_DEVIATION']):
            return vc.LOW_IMAGE_QUALITY_SMALL_ERROR, result_val
        else:
            return vc.GOOD_IMAGE, result_val
    
    def check_bright_or_dark(self, image_path=None, image=None, validation_cfg=vc.DEFAULT_VALIDATION_CONFIG):
        if image is None:
                image = cv2.imread(image_path)
                
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_part = cv2.inRange(gray, 0, 30)
        bright_part = cv2.inRange(gray, 220, 255)
        total_pixel = np.size(gray)
        dark_pixel = np.sum(dark_part > 0)
        bright_pixel = np.sum(bright_part > 0)
        
        dark_percent = dark_pixel/total_pixel
        bright_percent = bright_pixel/total_pixel
        
        result_val = {'dark_percent':dark_percent, 'bright_percent':bright_percent}
        
        if dark_percent > validation_cfg['DARK_THRESHOLD']:
            return vc.LOW_IMAGE_QUALITY_DARK_ERROR, result_val
        if bright_percent > validation_cfg['BRIGHT_THRESHOLD']:
            return vc.LOW_IMAGE_QUALITY_BRIGHT_ERROR, result_val
        else:
            return vc.GOOD_IMAGE, result_val

    def check_blurriness(self, image_path=None, image=None, validation_cfg=vc.DEFAULT_VALIDATION_CONFIG):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = image.shape
        # resize to fixed size
#        min_width = validation_cfg['MIN_WIDTH'] * validation_cfg['MIN_WIDTH_DEVIATION']
#        min_height = validation_cfg['MIN_HEIGHT'] * validation_cfg['MIN_HEIGHT_DEVIATION']
#        ratio = float(min_width*min_height) / float(image.shape[0] * image.shape[1])
        if h*w <= 800*600:
            ratio = float(800*600) / float(h*w)
        elif h*w <= 1024*768:
            ratio = float(1024*768) / float(h*w)
        elif h*w <= 1600*1200:
            ratio = float(1600*1200) / float(h*w)
        elif h*w <= 2000*2000:
            ratio = float(2000*2000) / float(h*w)
        else:
            ratio = float(3000*4000) / float(h*w)

        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)

        blur_map = cv2.Laplacian(image, cv2.CV_64F)
        score = np.var(blur_map)

        result_val = {'blurriness_score': score}

        if score > validation_cfg['BLURRINESS_THRESHOLD']:
            return vc.GOOD_IMAGE, result_val
        else:
            return vc.LOW_IMAGE_QUALITY_BLURRY_ERROR, result_val

    # def check_blurriness(self, image_path=None, image=None, validation_cfg=vc.DEFAULT_VALIDATION_CONFIG):
    #     # load the image, convert it to grayscale, and compute the
    #     # focus measure of the image using the Variance of Laplacian method
    #     if image is None:
    #         image = cv2.imread(image_path)
    #
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     blurriness = self.__variance_of_laplacian(gray)
    #
    #     result_val = {'blurriness': blurriness}
    #
    #     if blurriness > validation_cfg['BLURRINESS_THRESHOLD']:
    #         return vc.GOOD_IMAGE, result_val
    #     else:
    #         return vc.LOW_IMAGE_QUALITY_BLURRY_ERROR, result_val
    
    def check_histogram_is_normal(self, image_path=None, image=None, validation_cfg=vc.DEFAULT_VALIDATION_CONFIG):
        if image is None:
            image = cv2.imread(image_path, 0)

        his = cv2.calcHist([image], [0], None, [256], [0, 256])
        k2, p = stats.normaltest(his)

        result_val = {'k2_norm':k2, 'p_norm':p}

        if p < validation_cfg['NORMALITY_TEST_CHI_SQUARED_PROB_THRESHOLD']:
            return vc.LOW_IMAGE_QUALITY_BAD_ILLUMINATION_ERROR, result_val
        else:
            return vc.GOOD_IMAGE, result_val

    def validate_image(self, image=None, image_path=None, validation_cfg=vc.DEFAULT_VALIDATION_CONFIG):
        if validation_cfg is None:
            validation_cfg = vc.DEFAULT_VALIDATION_CONFIG
        elif validation_cfg is not None and validation_cfg != vc.DEFAULT_VALIDATION_CONFIG:
            cfg = vc.DEFAULT_VALIDATION_CONFIG.copy()
            cfg.update(validation_cfg)
            validation_cfg = cfg

        if image is None:
            image = cv2.imread(image_path)

        results = []
        for validator in self.get_validators():
            val_sw = create_elapsed_timer('sec')
            res, res_val = validator(image_path=image_path, image=image, validation_cfg=validation_cfg)

            logger.debug(
                'IN %s exec validator %s for %s with results %s, %s in %s',
                self.validate_image.__name__,
                validator.__name__,
                image_path,
                res,
                res_val,
                val_sw()
            )

            results.append(res_val)

            if res != vc.GOOD_IMAGE:
                return res, results

        return vc.GOOD_IMAGE, results
