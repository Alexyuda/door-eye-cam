from config import parser
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from pipeline.capture_video import CaptureVideo
from pipeline.detect_motion import DetectMotion
from pipeline.detect_faces import DetectFaces
from pipeline.display_video import DisplayVideo
from pipeline.annotate_image import AnnotateImage
from pipeline.record_video import RecordVideo
import threading
import time
from datetime import datetime

def set_interval(interval):
    def decorator(function):
        def wrapper(*args, **kwargs):
            stopped = threading.Event()

            def loop(): # executed in another thread
                while not stopped.wait(interval): # until stopped
                    function(*args, **kwargs)

            t = threading.Thread(target=loop)
            t.daemon = True # stop if the program exits
            t.start()
            return stopped
        return wrapper
    return decorator


@set_interval(3600)
def monitor_and_delete_old_videos(folder, delete_after_n_days):
    print('entered...')
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and ".mp4" in f]
    for file in files:
        last_updated_time = datetime.fromtimestamp(os.path.getmtime(file))
        dt_days = (datetime.now() - last_updated_time).total_seconds()/(24 * 60 * 60)
        if dt_days > delete_after_n_days:
            os.remove(file)


def main(args):
    monitor_and_delete_old_videos(args.vid_dump_dir, args.delete_after_n_days)

    capture_video = CaptureVideo(int(args.input) if args.input.isdigit() else args.input, args.select_roi)
    detect_motion = DetectMotion(args.p_min_motion_area)
    detect_faces = DetectFaces(prototxt=args.prototxt, model=args.model, confidence=args.confidence,
                               batch_size=args.batch_size)
    annotate_image = AnnotateImage("annotated_image") if args.display or args.out_video else None
    display_video = DisplayVideo("annotated_image") if args.display else None
    record_video = RecordVideo(args.vid_dump_dir, args.min_motion_to_save_video_sec, args.min_vid_length_sec)

    # Create image processing pipeline
    pipeline = (capture_video |
                detect_motion |
                detect_faces |
                annotate_image |
                display_video |
                record_video)

    # Iterate through pipeline
    progress = tqdm(total=capture_video.total_frame_count if capture_video.total_frame_count > 0 else None,
                    disable=not args.progress)

    try:
        for _ in pipeline:
            progress.update(1)
    except StopIteration:
        return
    except KeyboardInterrupt:
        return
    finally:
        progress.close()

        # Pipeline cleanup
        capture_video.cleanup()
        record_video.cleanup()
        if display_video:
            display_video.cleanup()


if __name__ == "__main__":
    args_ = parser.parse_args()
    main(args_)
