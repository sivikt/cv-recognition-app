import cv2
import numpy as np
import pathlib
import logging

from util.timer import create_elapsed_timer
import util.openvino as openvino_util
from . import car_validation as cv
from .car_validation import CarValidatorBase
from . import validation_consts as vc


logger = logging.getLogger(__name__)


class CarValidatorOpenVino(CarValidatorBase):
    def __init__(self, models_root_dir=pathlib.Path(__file__).parent):
        path_to_inference_graph = str(models_root_dir / 'models' / self.get_model_name() / 'openvino' / 'frozen_inference_graph.xml')
        path_to_inference_weights = str(models_root_dir / 'models' / self.get_model_name() / 'openvino' / 'frozen_inference_graph.bin')

        logger.info('LOADING PERSISTED MODELS FROM %s', path_to_inference_graph)

        exec_net, input_blob, out_blob, input_shape = openvino_util.load(path_to_inference_graph, path_to_inference_weights)

        logger.info('LOADED %s', path_to_inference_graph)

        self.net_exec = exec_net
        self.net_in_blob = input_blob
        self.net_out_blob = out_blob
        self.net_input_shape = input_shape

        path_to_labels = str(models_root_dir / 'models' / self.get_model_name() / 'classes-map.pbtxt')

    def __get_openvino_results(self, exec_net, out_blob, cur_request_id):
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            res = exec_net.requests[cur_request_id].outputs[out_blob]

            bboxes = []
            scores = []
            classes = []

            # The Inference Engine DetectionOutput layer consumes three tensors in the following order:
            #
            # Tensor with locations of bounding boxes
            # Tensor with confidences for each bounding box
            # Tensor with prior boxes (anchors in TensorFlow terminology)
            # DetectionOutput layer produces one tensor with seven numbers for each actual detection.
            # There are more output tensors in the TensorFlow Object Detection API,
            # but the values in them are consistent with the Inference Engine ones.
            for obj in res[0][0]:
                bboxes.append([obj[4], obj[3], obj[6], obj[5]])
                scores.append(obj[2])
                classes.append(obj[1])

            return np.array(bboxes), np.array(scores), np.array(classes).astype(np.uint8)
        else:
            raise Exception('Error during getting results for request ' + cur_request_id)

    def __run_cars_detection(self, image, net_exec, net_in_blob, net_out_blob, net_input_shape):
        inf_sw = create_elapsed_timer('sec')

        # Run inference
        n, c, h, w = net_input_shape

        in_frame = cv2.resize(image, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))

        net_exec.start_async(request_id=0, inputs={net_in_blob: in_frame})
        (bboxes, scores, classes) = self.__get_openvino_results(net_exec, net_out_blob, 0)

        # all outputs are float32 numpy arrays, so convert types as appropriate
        # output_dict['num_detections'] = int(output_dict['num_detections'][0])
        # output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        # output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        # output_dict['detection_scores'] = output_dict['detection_scores'][0]

        logger.debug('IN %s inference %s', self.__run_cars_detection.__name__, inf_sw())

        return bboxes, scores, classes

    def detect_cars_on_image(self, image_np, image_path=None, validation_cfg=vc.DEFAULT_VALIDATION_CONFIG):
        (orig_bboxes, orig_scores, orig_classes) = self.__run_cars_detection(image_np,
                                                                             self.net_exec,
                                                                             self.net_in_blob,
                                                                             self.net_out_blob,
                                                                             self.net_input_shape)

        detections = []

        for i in range(len(orig_classes)):
            if (orig_classes[i] == cv.MSCOCO_CATEGORY_INDEX['car']) and (orig_scores[i] > validation_cfg['CAR_SCORE_THRESHOLD']):
                detections.append({
                    'bbox': orig_bboxes[i],
                    'class': orig_classes[i],
                    'score': orig_scores[i]
                })

        return detections
