"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
"""

import os  # used to split the model location string
import cv2
import numpy as np
import logging as log
import math
from openvino.inference_engine import IENetwork, IECore  # used to load the IE python API


class GazeEstimationModel:
    """
            Class for the Face Detection Model.
    """

    def __init__(self, model_name, device='CPU', extensions=None):
        """
                This method intends to initialize all the attributes of the class
        :param model_name: The name of model .xml file
        :param device: Device Type(CPU/GPU/VPU/GPGA)
        :param extensions: CPU extension path
        """
        self.model = model_name
        self.device = device
        self.extension = extensions
        self.ie = None
        self.net = None
        self.inp = None
        self.out = None
        self.ext = None
        self.ex_net = None
        self.supported = None
        self.input_shape = None
        self.output_shape = None
        self.hpa = None

    def load_model(self):
        """
                This method is for loading the model to the device specified by the user.
                If your model requires any Plugins, this is where we can load them.
        :return: None
        """
        self.ie = IECore()
        model_xml = self.model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.net = IENetwork(model=model_xml, weights=model_bin)
        self.ex_net = self.ie.load_network(self.net)

        self.check_cpu_support()

        # to get the shape of input and output and set each to a class variable
        self.inp = next(iter(self.net.inputs))
        self.out = next(iter(self.net.outputs))

    def check_cpu_support(self):
        """
                This function intends to add the extension if CPU support is needed
        :return: None
        """
        unsupported = self.supported_layers
        if len(unsupported) != 0 and "CPU" in self.device:
            if self.extension is None:
                log.error("please provide the link to CPU extension, in order to run unsupported layers")
                exit(1)
            else:
                self.ie.add_extension(self.extension, "CPU")
                unsupported = self.supported_layers
                if len(unsupported) != 0:
                    log.error("Needs to exit, as some layers were unable to run on CPU as well")
                else:
                    exit(1)

    @property
    def supported_layers(self):
        """
                this function intends to find the unsupported layers on the given device
        :return: list of unsupported layers
        """
        self.supported = self.ie.query_network(network=self.net, device_name=self.device)
        unsupported_layers = [layer for layer in self.net.layers.keys() if layer not in self.supported]
        return unsupported_layers

    def get_input_shape(self):
        """
                This method intends to set the input shape parameter
        :return: None
        """
        self.input_shape = self.net.inputs[self.inp].shape

    def get_output_shape(self):
        """
                This method intends to set the input shape parameter
        :return: None
        """
        self.output_shape = self.net.outputs[self.out].shape

    def predict(self, image, hpa):
        """
        This method is meant for running predictions on the input image.
        """
        self.hpa = hpa
        left_proc, right_proc = self.preprocess_input(image[0].copy, image[1].copy)
        out_put = self.ex_net.infer(
            {'head_pose_angles': self.hpa, 'left_eye_image': left_proc, 'right_eye_image': right_proc})
        mouse_coord, gaze = self.preprocess_output(out_put)

        return mouse_coord, gaze

    def preprocess_input(self, left_image, right_image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        le_resized = cv2.resize(left_image, (self.input_shape[3], self.input_shape[2]))
        re_resized = cv2.resize(right_image, (self.input_shape[3], self.input_shape[2]))
        le_processed = np.transpose(np.expand_dims(le_resized, axis=0), (0, 3, 1, 2))
        re_processed = np.transpose(np.expand_dims(re_resized, axis=0), (0, 3, 1, 2))
        return le_processed, re_processed

    def preprocess_output(self, outputs):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        gaze_vector = outputs[self.out[0]].tolist()[0]
        rollv = self.hpa[2]  # angle_r_fc output from HeadPoseEstimation model
        cosv = math.cos(rollv * math.pi / 180.0)
        sinv = math.sin(rollv * math.pi / 180.0)

        newx = gaze_vector[0] * cosv + gaze_vector[1] * sinv
        newy = -gaze_vector[0] * sinv + gaze_vector[1] * cosv
        return (newx, newy), gaze_vector
