from typing import Optional
import inspect
from pathlib import Path
from utils import LOGGER, colorstr, ROOT
import time
import tqdm
import matplotlib.pyplot as plt



def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))

def monitor_queues(input_queue, inference_queue, output_queue):
    pbar = tqdm(total=0, bar_format='{desc}', leave=False)
    while True:
        input_queue_size = input_queue.qsize()
        inference_queue_size = inference_queue.qsize()
        output_queue_size = output_queue.qsize()

        current_time = time.time()

        pbar.desc = f"Input Queue Size: {input_queue_size}, Inference Queue Size: {inference_queue_size}, Output Queue Size: {output_queue_size}"

        # Calculate throughput for each module
        input_throughput = 1.0 / (current_time - input_queue.last_put_time)
        inference_throughput = 1.0 / (current_time - inference_queue.last_put_time)
        output_throughput = 1.0 / (current_time - output_queue.last_put_time)

        pbar.desc += f", Input Throughput: {input_throughput:.2f} fps, Inference Throughput: {inference_throughput:.2f} fps, Output Throughput: {output_throughput:.2f} fps"

        # Add any additional checks or monitoring logic as needed

        time.sleep(1)  # Monitor queues every 1 second

# Function to plot the queue sizes in horizontal bar charts
def plot_queue_sizes(input_queue, inference_queue, output_queue):
    while True:

        input_queue_size = input_queue.qsize()
        inference_queue_size = inference_queue.qsize()
        output_queue_size = output_queue.qsize()
        
        current_time = time.time()

        # Calculate throughput for each module
        input_throughput = 1.0 / (current_time - input_queue.last_put_time)
        inference_throughput = 1.0 / (current_time - inference_queue.last_put_time)
        output_throughput = 1.0 / (current_time - output_queue.last_put_time)

        print(f"\rInput Queue Size: {input_queue_size}, Inference Queue Size: {inference_queue_size}, Output Queue Size: {output_queue_size} Input Throughput: {input_throughput:.2f} fps, Inference Throughput: {inference_throughput:.2f} fps, Output Throughput: {output_throughput:.2f} fps", end="", flush=True)

        # Add any additional checks or monitoring logic as needed

        time.sleep(1)  # Plot charts every 1 second