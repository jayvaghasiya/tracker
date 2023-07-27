import cv2
from pypylon import pylon

from utils.profiler import Profile
from config import IS_NORMAL_CAMERA


class DataGenerator:
    def __init__(self, run_on_camera, input_video):
        self.run_on_camera = run_on_camera
        self.cam_time = Profile()

        if self.run_on_camera:
            if IS_NORMAL_CAMERA:
                self.camera = self.initialize_normal_camera()
            else:
                self.camera, self.converter = self.initialize_camera()
        else:
            self.vidcap = cv2.VideoCapture(input_video)
            print(input_video, self.vidcap)

    @staticmethod
    def initialize_camera():
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        return camera, converter

    @staticmethod
    def initialize_normal_camera():
        vid = cv2.VideoCapture(0)
        return vid

    def get_next_frame(self):
        with self.cam_time:
            if self.run_on_camera:
                if IS_NORMAL_CAMERA:
                    ret, frame = self.camera.read()
                    return frame
                else:
                    grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                    if grabResult.GrabSucceeded():
                        image = self.converter.Convert(grabResult)
                        return image.GetArray()
                    else:
                        return None
            else:
                ret, frame = self.vidcap.read()
                if ret:
                    return frame
                else:
                    return None
