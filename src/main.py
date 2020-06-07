import os
import cv2
import logging as log
from argparse import ArgumentParser
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandMarkDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel
from mouse_controller import MouseController
from input_feeder import InputFeeder


def build_argparser():
    """
            Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-f", "--face_detection_model", required=True, type=str,
                        help="Path to an .xml file with Face Detection model.")
    parser.add_argument("-fl", "--facial_landmark_model", required=True, type=str,
                        help="Path to an .xml file with Facial Landmark Detection model.")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str,
                        help="Path to an .xml file with Head Pose Estimation model.")
    parser.add_argument("-g", "--gaze_estimation_model", required=True, type=str,
                        help="Path to an .xml file with Gaze Estimation model.")

    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or CAM")

    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")

    parser.add_argument("-pt", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for detection fitering.")

    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help='Specify the target device to infer on: '
                             'CPU, GPU, FPGA or MYRIAD is acceptable. Sample '
                             'will look for a suitable plugin for device '
                             'specified (CPU by default)')

    return parser


def check(args):
    """
            This function will return the InputFeeder object
    :param args:
    :return: InputFeeder object
    """
    if args.input == 'CAM':
        in_put_feeder = InputFeeder('CAM')
    else:
        if not os.path.isfile(args.input):
            log.error("Unable to locate the file")
            exit(1)
        in_put_feeder = InputFeeder("video",args.input)
    return in_put_feeder


def check_input_files(file):
    """
            This function check if the model files exists or not
    :param file:
    :return: 1 if file is not found
    """
    if not os.path.isfile(file):
        print(file + " was not able to load")
        return 1
    else:
        return 0    


def loader(in_put_feeder, fdm, fldm, gem, hpem):
    """
            This function is intended to load the model files
    :param in_put_feeder: InputFeeder object
    :param fdm: FaceDetectionModel object
    :param fldm: FacialLandMarkDetectionModel object
    :param gem: GazeEstimationModel object
    :param hpem: HeadPoseEstimationModel object
    :return: None
    """
    in_put_feeder.load_data()
    fdm.load_model()
    fldm.load_model()
    hpem.load_model()
    gem.load_model()


def main():
    """
            This is the control function of the whole project
    :return: None
    """
    global fdm, fldm, gem, hpem

    args = build_argparser().parse_args()

    in_put_feeder = check(args)

    if check_input_files(args.face_detection_model) == 0:
        fdm = FaceDetectionModel(args.face_detection_model, args.device, args.cpu_extension, args.prob_threshold)
    if check_input_files(args.facial_landmark_model) == 0:
        fldm = FacialLandMarkDetectionModel(args.facial_landmark_model, args.device, args.cpu_extension)
    if check_input_files(args.gaze_estimation_model) == 0:
        gem = GazeEstimationModel(args.gaze_estimation_model, args.device, args.cpu_extension)
    if check_input_files(args.head_pose_model) == 0:
        hpem = HeadPoseEstimationModel(args.head_pose_model, args.device, args.cpu_extension)

    mouse = MouseController('medium', 'fast')

    loader(in_put_feeder, fdm, fldm, gem, hpem)

    frame_c = 0

    for ret, frame in in_put_feeder.next_batch():
        if not ret:
            break
        frame_c += 1
        if frame_c % 5 == 0:
            cv2.imshow('Video', cv2.resize(frame, (500, 500)))

        key = cv2.waitKey(60)

        face, control = fdm.predict(frame.copy())
        if type(face) == int:
            log.error("Unable to detect face")
            if key == 27:
                break
            continue

        hpe_out = hpem.predict(face.copy())
        left, right, eye_coords = fldm.predict(face.copy())
        mouse_vec, gaze_vec = gem.predict(left, right, hpe_out)

        if frame_c % 5 == 0:
            mouse.move(mouse_vec[0], mouse_vec[1])
        if key == 27:
            break

    cv2.destroyAllWindows()
    in_put_feeder.close()


if __name__ == "__main__":
    main()
