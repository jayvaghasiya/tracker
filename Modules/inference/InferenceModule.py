import multiprocessing as mp
import time

import cv2
import numpy as np
import torch

from Modules.inference.inference_lib import TRT_engine
from config import BATCH_SIZE


class BatchInferenceModule(mp.Process):
    def __init__(self, input_queue, inference_queue, engine_path):
        super(BatchInferenceModule, self).__init__()
        self.imgsz = [640, 640]
        self.device = torch.device('cuda:0')
        self.input_queue = input_queue
        self.inference_queue = inference_queue
        self.engine_path = engine_path

    def letterbox(self, im, color=(114, 114, 114), auto=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.imgsz
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, r, dw, dh

    def preprocess(self, image):
        img, self.r, self.dw, self.dh = self.letterbox(image)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        return img

    def run(self):
        inference_engine = TRT_engine(self.engine_path)
        threshold = 0.5
        while True:
            batch = []
            orig_batch = []
            while len(batch) < BATCH_SIZE:
                if not self.input_queue.empty():
                    orig_batch.append(self.input_queue.get())
                    batch.append(self.preprocess(orig_batch[-1]))
                else:
                    time.sleep(0.001)  # Sleep briefly to avoid busy waiting

            if len(batch) > 0:
                [nums, boxes, scores, classes] = inference_engine.predict(torch.stack(batch))
                for frame_id in range(BATCH_SIZE):
                    num = int(nums[frame_id][0])
                    new_bboxes = []
                    for i in range(num):
                        if scores[frame_id][i] < threshold:
                            continue
                        xmin = (boxes[frame_id][i][0] - self.dw) / self.r
                        ymin = (boxes[frame_id][i][1] - self.dh) / self.r
                        xmax = (boxes[frame_id][i][2] - self.dw) / self.r
                        ymax = (boxes[frame_id][i][3] - self.dh) / self.r
                        new_bboxes.append([classes[frame_id][i], scores[frame_id][i], xmin, ymin, xmax, ymax])
                    self.inference_queue.put([orig_batch[frame_id], new_bboxes])