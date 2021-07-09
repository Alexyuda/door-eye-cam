import cv2
import os
from pipeline.pipeline import Pipeline
from datetime import datetime


class RecordVideo(Pipeline):
    """Pipeline task to save video that contain motion."""

    codec = 'avc1'
    video_dump_fn = None
    start_record_time = None
    is_recording = None
    cap = None

    def __init__(self, src, dump_dir, min_motion_to_save_video_sec=0.1, min_vid_length_sec=30,
                 close_cap_and_start_new_one_after_n_secs=None):
        self.src = src
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        self.dump_dir = dump_dir
        self.min_motion_to_save_video_sec = min_motion_to_save_video_sec
        self.min_vid_length_sec = min_vid_length_sec
        self.close_cap_and_start_new_one_after_n_secs = close_cap_and_start_new_one_after_n_secs
        self.motion_counter_frames = 0

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

                # start recording if enough frames include movement
                if self.motion_counter_frames / data["fps"] > self.min_motion_to_save_video_sec and not self.cap:
                    self.video_dump_fn = os.path.join(self.dump_dir, data['time'].strftime("%Y-%m-%d_%H-%M-%S")+".mp4")
                    self.is_recording = True
                    self.start_record_time = data['time']
                    self.cap = cv2.VideoWriter(self.video_dump_fn,
                                               cv2.VideoWriter_fourcc(*self.codec),
                                               data["fps"],
                                               (data["image"].shape[1], data["image"].shape[0]))

                # break long videos
                if self.is_recording and self.close_cap_and_start_new_one_after_n_secs is not None:
                    now_time = datetime.now()
                    if (now_time - self.start_record_time).total_seconds() > \
                            self.close_cap_and_start_new_one_after_n_secs:
                        self.cap.release()
                        self.video_dump_fn = os.path.join(self.dump_dir,
                                                          now_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4")
                        self.cap = cv2.VideoWriter(self.video_dump_fn,
                                                   cv2.VideoWriter_fourcc(*self.codec),
                                                   data["fps"],
                                                   (data["image"].shape[1], data["image"].shape[0]))
                        self.start_record_time = now_time

                # end recording if enough time passed without movement
                if self.is_recording:
                    if (data['time'] - self.start_record_time).total_seconds() > self.min_vid_length_sec \
                            and self.motion_counter_frames == 0:
                        self.cleanup()
                        self.is_recording = False
                        self.video_dump_fn = None
                        self.start_record_time = None
                        self.cap = None
                    else:
                        self.cap.write(data[self.src])

                if self.filter(data):
                    yield self.map(data)

            except StopIteration:
                return

    def cleanup(self):
        if self.cap:
            self.cap.release()
