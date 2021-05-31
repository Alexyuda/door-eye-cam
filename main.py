from config import parser
import os
from tqdm import tqdm
from pipeline.capture_video import CaptureVideo
from pipeline.detect_motion import DetectMotion
from pipeline.detect_faces import DetectFaces
from pipeline.display_video import DisplayVideo
from pipeline.annotate_image import AnnotateImage
from pipeline.record_video import RecordVideo


def main(args):
    capture_video = CaptureVideo(int(args.input) if args.input.isdigit() else args.input)
    detect_motion = DetectMotion(args.p_min_motion_area)
    detect_faces = DetectFaces(prototxt=args.prototxt, model=args.model, confidence=args.confidence, batch_size=args.batch_size)
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
    progress = tqdm(total=capture_video.frame_count if capture_video.frame_count > 0 else None,
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
    args = parser.parse_args()
    main(args)
