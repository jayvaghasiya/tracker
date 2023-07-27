# Define the number of processes/threads for each module (adjust according to your hardware)
NUM_INPUT_SOURCE_PROCESSES = 1
NUM_BATCH_INFERENCE_PROCESSES = 1
NUM_OBJECT_TRACKING_PROCESSES = 1

# Define the size of the buffer/queue for each module
INPUT_SOURCE_BUFFER_SIZE = 550
BATCH_INFERENCE_BUFFER_SIZE = 50
OBJECT_TRACKING_BUFFER_SIZE = 50

# Inference Module batch size
BATCH_SIZE = 8

# Input source Calibration
IS_NORMAL_CAMERA = True
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes