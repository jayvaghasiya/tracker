import contextlib
import math
from pathlib import Path
from urllib.error import URLError

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils import FONT, USER_CONFIG_DIR, threaded

from .checks import check_font, check_requirements, is_ascii
from .files import increment_path
from .ops import clip_coords, xywh2xyxy, xyxy2xywh


class Colors:
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


class Annotator:
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc', list_of_points=None):
        self.im = im
        self.imc = im.copy()
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

        # White Boundary Box for ROI
        #for i in range(len(list_of_points)):
        #    self.im = cv2.line(self.im, (list_of_points[i - 1][0], list_of_points[i - 1][1]),
        #                       (list_of_points[i][0], list_of_points[i][1]), (255, 255, 255), self.lw)

    def dump_frame_info(self, frame_idx):
        tf = max(self.lw, 1)
        label = "Frame : {}".format(frame_idx)
        w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]
        p1 = (0, 0)
        h = h + 15
        w = w + 6
        cv2.rectangle(self.im, (0, 0), (w, h - 5), (255, 255, 255), -1, cv2.LINE_AA)  # filled
        cv2.putText(self.im,
                    label, (p1[0], p1[1] + h - 8),
                    0,
                    self.lw / 3,
                    (128, 128, 128),
                    thickness=tf,
                    lineType=cv2.LINE_AA)

    def add_picker(self, xyxys, color=(51, 51, 255)):
        hand_crops = []
        for array in xyxys:
            array = array.astype(int)
            p1, p2 = [int(array[0])-int(array[2])//2, int(array[1])-int(array[3])//2], [int(array[0])+int(array[2])//2, int(array[1])+int(array[3])//2]
            if p1[0] < 0:
                p1[0] = 0
            if p2[0] < 0:
                p2[0] = 0
            if p1[1] < 0:
                p1[1] = 0
            if p2[1] < 0:
                p2[1] = 0
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw - 1, lineType=cv2.LINE_AA)
            cv2.circle(self.im, (array[0], array[1]+array[3]//3), 300, (51, 51, 255), 1)
            hand_crops.append(self.imc[p1[1]:p2[1], p1[0]:p2[0]])
        return hand_crops

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), trajectory=[], is_picked=False):
        """Add one xyxy box to image with label"""

        # Label settings
        label = label.upper()
        tf = max(self.lw - 1, 1) * 2  # Font thickness (integer)
        tf = max(tf - 2, 1)
        fs = self.lw / 2.8            # Font scale (float, multiplier on base size)
        margin = 6                    # Margin of background around text (pixels)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Rectangle for object
        pp1, pp2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, pp1, pp2, color, thickness=self.lw - 1, lineType=cv2.LINE_AA)

        # Rectangle for text background
        w, h = cv2.getTextSize(label, font, fontScale=fs, thickness=tf)[0]  # text width, height
        outside = pp1[1] - h >= 3
        p1 = pp1[0] - margin, pp1[1] + margin
        p2 = pp1[0] + w + margin, pp1[1] - h - 3 - margin if outside else pp1[1] + h + 3 - margin
        cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled

        # Label text
        cv2.putText(self.im,
                    label, (pp1[0], pp1[1] - 2 if outside else pp1[1] + h + 2),
                    font,
                    fs,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)

        if is_picked:
            image_with_box = self.im.copy()
            image_with_box = cv2.rectangle(image_with_box, pp1, pp2, color, thickness=-1)
            self.im = cv2.addWeighted(self.im, 0.5, image_with_box, 0.5, 0)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor='top'):
        # Add text to image (PIL-only)
        if anchor == 'bottom':  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        self.draw.text(xy, text, fill=txt_color, font=self.font)

    def fromarray(self, im):
        # Update self.im from a numpy array
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)
