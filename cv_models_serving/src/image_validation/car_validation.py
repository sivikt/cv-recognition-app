import cv2
import numpy as np
import pathlib
import tensorflow as tf
import math
import logging

from ..third_party import label_map_util as label_map_util
from ..util.timer import create_elapsed_timer
from . import validation_consts as vc


logger = logging.getLogger(__name__)


root_dir = pathlib.Path(__file__).parent
MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28'
PATH_TO_LABELS = str(root_dir / 'models' / MODEL_NAME / 'mscoco_label_map.pbtxt')


def get_classes_index():
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    category_invindex = {}

    for k, v in category_index.items():
        category_invindex[v['name']] = k

    return category_invindex


MSCOCO_CATEGORY_INDEX = get_classes_index()


class CarValidatorBase:
    def __init__(self, models_root_dir=pathlib.Path(__file__).parent, intra_op=None, inter_op=None):
        path_to_frozen_graph = str(models_root_dir / 'models' / self.get_model_name() / 'frozen_inference_graph.pb')
        
        logger.info('LOADING PERSISTED MODELS FROM %s', str(path_to_frozen_graph))
    
        self.detection_graph = tf.Graph()

        if intra_op is not None and inter_op is not None:
            logger.info('USE intra_op=%s, inter_op=%s for %s', intra_op, inter_op, str(path_to_frozen_graph))

            config = tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=intra_op,
                inter_op_parallelism_threads=inter_op,
                allow_soft_placement=True,
                gpu_options={'allow_growth': True},
                device_count={'CPU': intra_op}
            )
        else:
            config = tf.compat.v1.ConfigProto(gpu_options={'allow_growth': True})

        config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

        with self.detection_graph.as_default() as graph:
            with tf.io.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def = tf.compat.v1.GraphDef()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

                # Get handles to input and output tensors
                ops = graph.get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                det_tensor_dict = {}

                for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        det_tensor_dict[key] = graph.get_tensor_by_name(tensor_name)

                det_image_tensor = graph.get_tensor_by_name('image_tensor:0')

        self.detection_session = tf.compat.v1.Session(config=config, graph=self.detection_graph)
        self.detection_tensor_dict = det_tensor_dict
        self.detection_image_tensor = det_image_tensor

        # for op in self.detection_graph.get_operations():
        #     in_shapes = [str(tf.shape(i)) for i in op.inputs]
        #     logger.info('{:<50} {:<20} {:<10}'.format(str(op.node_def.name), str(in_shapes), str(op.type)))
        #     logger.info('-' * 80)

        logger.info('LOADED %s', str(path_to_frozen_graph))

    def get_model_name(self):
        return MODEL_NAME

    def __run_inference_for_single_image(self, image, session, tensor_dict, image_tensor):
        load_sw = create_elapsed_timer('sec')

        logger.debug('IN %s session, init in %s', self.__run_inference_for_single_image.__name__, load_sw())

        inf_sw = create_elapsed_timer('sec')

        # Run inference
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        output_dict = session.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        logger.debug('IN %s inference in %s', self.__run_inference_for_single_image.__name__, inf_sw())
    
        return output_dict
    
    def detect_cars_on_image(self, image_np, image_path=None, validation_cfg=vc.DEFAULT_VALIDATION_CONFIG):
        output_dict = self.__run_inference_for_single_image(image_np,
                                                            self.detection_session, 
                                                            self.detection_tensor_dict, 
                                                            self.detection_image_tensor)
    
        orig_boxes = output_dict['detection_boxes']
        orig_classes = output_dict['detection_classes']
        orig_scores = output_dict['detection_scores']

        orig_scores_nonzero = (orig_scores > 0.).nonzero()
        orig_boxes = orig_boxes[orig_scores_nonzero]
        orig_classes = orig_classes[orig_scores_nonzero]
        orig_scores = orig_scores[orig_scores_nonzero]
    
        detections = []

        logger.debug('IN %s %s %s %s', self.detect_cars_on_image.__name__, image_path, orig_classes, orig_scores)

        for i in range(len(orig_classes)):
            if (orig_classes[i] == MSCOCO_CATEGORY_INDEX['car']) or (orig_classes[i] == MSCOCO_CATEGORY_INDEX['motorcycle']) \
                or (orig_classes[i] == MSCOCO_CATEGORY_INDEX['bus']) or (orig_classes[i] == MSCOCO_CATEGORY_INDEX['truck']):

                logger.debug('IN %s %s %s %s', self.detect_cars_on_image.__name__, image_path, orig_classes[i], orig_scores[i])
                if orig_scores[i] > validation_cfg['CAR_SCORE_THRESHOLD']:
                    detections.append({
                        'bbox': orig_boxes[i],
                        'class': orig_classes[i],
                        'score': orig_scores[i]
                    })
    
        return detections
    
    def check_car_is_on_image(self, image_path=None, image=None, validation_cfg=vc.DEFAULT_VALIDATION_CONFIG):
        if image is None:
            image = cv2.imread(image_path)
    
        height, width, depth = image.shape
        img_area = height * width
    
        detections = self.detect_cars_on_image(image, image_path, validation_cfg)
    
        result_val = {'cars': detections}
    
        ## check for CAR_IMPROPER_POSITION_ERROR and TOO_MANY_CARS_ERROR
        if len(detections) == 0:
            return vc.LOW_CAR_QUALITY_NO_CAR_ERROR, result_val
        else:
            total_sw = create_elapsed_timer('sec')
    
            try:
                x_center = width/2
                y_center = height/2
                circumcircle_radius = math.sqrt(x_center**2 + y_center**2)
    
    
                cars_area = []
                for det in detections:
                    # bboxes as (ymin, xmin, ymax, xmax)
                    area = float((det['bbox'][2]-det['bbox'][0]) * height * (det['bbox'][3]-det['bbox'][1]) * width)
                    cars_area.append(area)
    
    
                sorted_cars_area = np.argsort(cars_area)
    
                car_area_ratio = cars_area[sorted_cars_area[-1]] / img_area
                biggest_car_area_ratio = (cars_area[sorted_cars_area[-1]] / cars_area[sorted_cars_area[-2]]) if len(detections) > 1 else math.inf
    
                (car_ymin, car_xmin, car_ymax, car_xmax) = detections[sorted_cars_area[-1]]['bbox']
                car_x_center = (car_xmin + car_xmax)*x_center
                car_y_center = (car_ymin + car_ymax)*y_center
                car_radius_vec = math.sqrt((x_center-car_x_center)**2 + (y_center-car_y_center)**2)
                car_radius_vec_ratio = car_radius_vec / circumcircle_radius
    
                result_val['car_area_ratio'] = car_area_ratio
                result_val['biggest_car_area_ratio'] = biggest_car_area_ratio
                result_val['car_radius_vec_ratio'] = car_radius_vec_ratio
    
                if car_area_ratio < validation_cfg['CAR_MIN_AREA_THRESHOLD']:
                    return vc.LOW_CAR_QUALITY_SMALL_CAR_ERROR, result_val
                elif car_area_ratio < 0.6 and car_radius_vec_ratio > validation_cfg['CAR_CENTER_DEVIATION_THRESHOLD']:
                    return vc.LOW_CAR_QUALITY_NOT_CENTERED_CAR_ERROR, result_val
                elif biggest_car_area_ratio < validation_cfg['TWO_BIGGEST_CARS_RATIO_THRESHOLD']:
                    return vc.LOW_CAR_QUALITY_CAR_AMBIGUITY_ERROR, result_val
                else:
                    return vc.GOOD_IMAGE, result_val
            finally:
                logger.debug('IN %s finished other car checks in %s', self.check_car_is_on_image.__name__, total_sw())
