from pathlib import Path
import multiprocessing as mp

from config import VID_FORMATS
from Modules.datageneration.data_generator import DataGenerator


class InputSourceModule(mp.Process):
    def __init__(self, input_queue, source):
        super(InputSourceModule, self).__init__()
        self.input_queue = input_queue
        self.source = str(source)

    def run(self):
        # Your code to capture frames from the input source (e.g., camera or video file)
        # Put frames into the input queue

        if Path(self.source).suffix[1:] in VID_FORMATS:
            datagenerator = DataGenerator(False, self.source)
            frame = datagenerator.get_next_frame()
            self.input_queue.put(frame)
        else:
            datagenerator = DataGenerator(True, None)

        while True:
            frame = datagenerator.get_next_frame()
            if frame is None:
                break
            self.input_queue.put(frame)
        print("Exiting InputSourceModule !!")
