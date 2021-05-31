import cv2
import os
from pipeline.pipeline import Pipeline


class RecordVideo(Pipeline):
    """Pipeline task to save video that contain motion."""

    motion_counter_frames = 0
    codec = 'avc1'
    video_dump_fn = None
    start_record_time = None
    is_recording = None
    cap = None

    def __init__(self, dump_dir, min_motion_to_save_video_sec=0.1, min_vid_length_sec=30):
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        self.dump_dir = dump_dir
        self.min_motion_to_save_video_sec = min_motion_to_save_video_sec
        self.min_vid_length_sec = min_vid_length_sec

        super(RecordVideo, self).__init__()

    def generator(self):

        while self.has_next():
            try:
                data = next(self.source)
                is_motion_in_frame = len(data["motion_bboxes"]) > 0
                if is_motion_in_frame:
                    self.motion_counter_frames += 1
                else:
                    self.motion_counter_frames -= 1
                    self.motion_counter_frames = max(self.motion_counter_frames, 0)

                motion_counter_sec = self.motion_counter_frames / data["fps"]

                # start recording if enough frames include movement
                if motion_counter_sec > self.min_motion_to_save_video_sec and not self.cap:
                    self.video_dump_fn = os.path.join(self.dump_dir, data['time'].strftime("%Y-%m-%d_%H-%M-%S")+".mp4")
                    self.is_recording = True
                    self.motion_counter_frames = 0
                    self.start_record_time = data['time']
                    self.cap = cv2.VideoWriter(self.video_dump_fn,
                                               cv2.VideoWriter_fourcc(*self.codec),
                                               data["fps"],
                                               (data["image"].shape[1], data["image"].shape[0]))

                # end recording if enough time passed without movement
                if self.is_recording:
                    if (data['time'] - self.start_record_time).total_seconds() > self.min_vid_length_sec \
                            and self.motion_counter_frames == 0:
                        self.cleanup()
                        motion_counter_sec = 0
                        self.is_recording = False
                        self.video_dump_fn = None
                        self.start_record_time = None
                        self.cap = None
                    else:
                        self.cap.write(data["annotated_image"])

                if self.filter(data):
                    yield self.map(data)

            except StopIteration:
                return

    def cleanup(self):
        if self.cap:
            self.cap.release()
