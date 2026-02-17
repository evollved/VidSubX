import os
from unittest import TestCase

from infra.app_paths import AppPaths
from shared.utils import video_details, default_sub_area, frame_no_to_duration, timecode

ch_vid = "test data/chinese_vid.mp4"
os.chdir(AppPaths.working_dir)


class TestUtils(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("\nRunning setUpClass method...")
        cls.fps, cls.frame_total, cls.frame_width, cls.frame_height = 30.0, 1830, 1920, 1080

    def test_video_details(self):
        print("\nRunning tests for video_details method...")
        fps, frame_total, frame_width, frame_height = video_details(ch_vid)
        self.assertEqual(fps, self.fps)
        self.assertEqual(frame_total, self.frame_total)
        self.assertEqual(frame_width, self.frame_width)
        self.assertEqual(frame_height, self.frame_height)

    def test_default_sub_area(self):
        print("\nRunning tests for default_sub_area method...")
        x1, y1, x2, y2 = default_sub_area(self.frame_width, self.frame_height)
        self.assertEqual(x1, 0)
        self.assertEqual(y1, 810)
        self.assertEqual(x2, self.frame_width)
        self.assertEqual(y2, self.frame_height)

    def test_frame_no_to_duration(self):
        print("\nRunning tests for frame_no_to_duration method...")
        self.assertEqual(frame_no_to_duration(457, 30.0), "00:00:15:233")
        self.assertEqual(frame_no_to_duration(915, 30.0), "00:00:30:500")
        self.assertEqual(frame_no_to_duration(369, 24.0), "00:00:15:375")
        self.assertEqual(frame_no_to_duration(739, 24.0), "00:00:30:791")

    def test_timecode(self):
        print("\nRunning tests for timecode method...")
        self.assertEqual(timecode(4577987976), "1271:39:47,976")
        self.assertEqual(timecode(97879869), "27:11:19,869")
        self.assertEqual(timecode(309485036), "85:58:05,036")
        self.assertEqual(timecode(378786979), "105:13:06,979")
        self.assertEqual(timecode(25234.7962452), "00:00:25,234")
        self.assertEqual(timecode(6365.242454), "00:00:06,365")
