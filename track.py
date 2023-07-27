import argparse
from pathlib import Path

from Modules.ObjectTrackingModule import ObjectTrackingModule
from Modules.VisualizationModule import VisualizationModule
from Modules.datageneration.InputSourceModule import InputSourceModule
from Modules.inference.InferenceModule import BatchInferenceModule
from config import INPUT_SOURCE_BUFFER_SIZE, BATCH_INFERENCE_BUFFER_SIZE, OBJECT_TRACKING_BUFFER_SIZE
from utils.MonitoredQueue import MonitoredQueue
from utils.utils import print_args, plot_queue_sizes

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def run(
        source='0',
        engine='waste-management.trt',
        tracking_method='bytetrack',
        tracking_config=None,
        roi_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='Waste-Management-Tracker',  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        no_laser_send=True,
        laser_record=None,  # record a session
        stream_video=False,
        save_and_upload=False,
        archive=None,
        config_file=None,
        mem_limit=None,
        show_fps=False
):
    # Create queues for communication between modules
    input_queue = MonitoredQueue(maxsize=INPUT_SOURCE_BUFFER_SIZE)
    inference_queue = MonitoredQueue(maxsize=BATCH_INFERENCE_BUFFER_SIZE)
    output_queue = MonitoredQueue(maxsize=OBJECT_TRACKING_BUFFER_SIZE)

    # Create and start the processes for each module
    input_process = InputSourceModule(input_queue, source=source)
    inference_process = BatchInferenceModule(input_queue, inference_queue, engine_path=engine)
    tracking_process = ObjectTrackingModule(inference_queue, output_queue, tracking_method, tracking_config, roi_config)

    # Optional: Create and start the visualization process
    visualization_process = VisualizationModule(output_queue)

    process_list = [input_process, inference_process, tracking_process, visualization_process]
    try:
        for process in process_list:
            process.start()

        plot_queue_sizes(input_queue, inference_queue, output_queue)

    except KeyboardInterrupt:
        # Catch KeyboardInterrupt (Ctrl+C) to stop the processes gracefully
        print("KeyboardInterrupt: Stopping all processes...")

    finally:
        # Terminate all processes to ensure clean exit
        for process in process_list:
            process.terminate()

        # Wait for all processes to finish
        for process in process_list:
            process.join()

    print("All processes have been stopped.")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--tracking-method', type=str, default='bytetrack', help='botsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--no_laser_send', default=True, action='store_false', help='Send output data to laser repo')
    parser.add_argument('--laser_record', default=None, help='Do a recorded session')
    parser.add_argument('--stream_video', default=False, action='store_true', help='stream video data to aws')
    parser.add_argument('--save_and_upload', default=False, action='store_true', help='stream video data to aws s3')
    parser.add_argument('--archive', type=str, default='/media/arthur/Sorted')
    parser.add_argument('--config_file', type=Path,
                        default=Path('~/laser/daemon/etc/apiserver-config.yaml').expanduser())
    parser.add_argument('--mem-limit', type=int,
                        help='Limits VIRT RAM usage to this value, in megabytes. Not available on Windows.')
    parser.add_argument('--show_fps', default=False, action='store_true', help='if true, show FPS on video')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'boxmot' / 'configs' / (opt.tracking_method + '.yaml')
    opt.roi_config = opt.config_file
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
