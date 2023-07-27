import multiprocessing as mp
from boxmot.tracker_zoo import create_tracker
import time
import cv2


class ObjectTrackingModule(mp.Process):
    def __init__(self, inference_queue, output_queue, tracking_method, tracking_config, roi_config):
        super(ObjectTrackingModule, self).__init__()
        self.inference_queue = inference_queue
        self.output_queue = output_queue
        self.tracking_method = tracking_method
        self.tracking_config = tracking_config
        self.roi_config = roi_config

    @staticmethod
    def visualize(img, bbox_array):
        for temp in bbox_array:
            xmin = int(temp[2])
            ymin = int(temp[3])
            xmax = int(temp[4])
            ymax = int(temp[5])
            clas = int(temp[0])
            score = temp[1]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (105, 237, 249), 2)
            img = cv2.putText(img, "class:" + str(clas) + " " + str(round(score, 2)), (xmin, int(ymin) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 1)
        return img

    def run(self):
        # print("HERE !!@#$%$")
        tracker, list_of_points = create_tracker(
            self.tracking_method,
            self.tracking_config,
            self.roi_config
        )

        # Variables to calculate FPS
        fps_avg_frame_count = 10
        counter, fps = 0, 0
        start_time = time.time()
        while True:
            if not self.inference_queue.empty():
                img, bboxes = self.inference_queue.get()
                print(bboxes)
                tracker.update()
                # counter += 1
                # if counter % fps_avg_frame_count == 0:
                #     end_time = time.time()
                #     fps = fps_avg_frame_count / (end_time - start_time)
                #     start_time = time.time()
                # # Show the FPS
                # fps_text = 'FPS = {:.1f}'.format(fps)
                # print(fps_text)
                # img = self.visualize(img, bboxes)
                # outfile = '%s/%s.jpg' % ("./output", "out_" + str(uuid.uuid4()))
                # print(outfile)
                # cv2.imwrite(outfile, img)
                # print(detections)
                # tracked_objects = ...  # Perform object tracking on the detections
                # self.output_queue.put(tracked_objects)
            else:
                time.sleep(0.001)
