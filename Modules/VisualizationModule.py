import multiprocessing as mp


class VisualizationModule(mp.Process):
    def __init__(self, output_queue):
        super(VisualizationModule, self).__init__()
        self.output_queue = output_queue

    def run(self):
        # Your code to run batch inference using TensorRT
        while True:
            # Dosomething
            pass