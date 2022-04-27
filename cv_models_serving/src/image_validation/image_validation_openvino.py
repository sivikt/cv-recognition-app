from .car_validation_openvino import CarValidatorOpenVino
from .image_validation import ValidationBase


class ValidationOpenVino(ValidationBase):

    def __init__(self):
        self.cars_validator = CarValidatorOpenVino()
