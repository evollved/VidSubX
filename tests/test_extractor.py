import os
from pathlib import Path
from unittest import TestCase

from infra.app_paths import AppPaths

os.chdir(AppPaths.working_dir)

config_file = Path("config.ini")
config_file.unlink(missing_ok=True)  # The tests only work on the default config

from main import SubtitleExtractor, setup_ocr

ch_vid = "test data/chinese_vid.mp4"
ch_vid_srt = Path("test data/chinese_vid.srt")
setup_ocr()


class TestSubtitleExtractor(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("\nRunning setUpClass method...")
        cls.se = SubtitleExtractor()
        cls.fps, cls.frame_total, cls.frame_width, cls.frame_height = 30.0, 1830, 1920, 1080
        cls.default_sub_area = 0, 810, 1920, 1080

    def test_similarity(self):
        print("\nRunning tests for similarity method...")
        self.assertEqual(self.se.similarity("这漫天的星辰之中", "竟然还蕴含着星辰之力"), 0.3333333333333333)
        self.assertEqual(self.se.similarity("竟然还蕴含着星辰之力", "竟竞然还蕴含着星辰之力"), 0.9523809523809523)
        self.assertEqual(self.se.similarity("此机会多吸取一点", "大胆人类"), 0.0)
        self.assertEqual(self.se.similarity("颗果实就想打发我们", "颗果实就想打发我们"), 1.0)

    def test_similar_text_name_gen(self):
        print("\nRunning tests for similar_text_name_gen method...")
        start_name, end_name = "3466.666666666667--4733.333333333333", "3466.666666666667--4733.333333333333"
        self.assertEqual(self.se.similar_text_name_gen(start_name, end_name), "3466.666666666667--4733.333333333333")
        start_name, end_name = "5066.666666666666--6200.0", "6333.333333333333--6666.666666666667"
        self.assertEqual(self.se.similar_text_name_gen(start_name, end_name), "5066.666666666666--6666.666666666667")
        start_name, end_name = "9866.666666666668--10466.666666666666", "11200.0--11733.333333333332"
        self.assertEqual(self.se.similar_text_name_gen(start_name, end_name), "9866.666666666668--11733.333333333332")
        start_name, end_name = "59733.333333333336--59800.0", "60933.33333333333--60933.33333333333"
        self.assertEqual(self.se.similar_text_name_gen(start_name, end_name), "59733.333333333336--60933.33333333333")

    def test_name_to_duration(self):
        print("\nRunning tests for name_to_duration method...")
        self.assertEqual(self.se.name_to_duration("5066.666666666666--6666.666666666667"), 1600.000000000001)
        self.assertEqual(self.se.name_to_duration("17800.0--18200.0"), 400.0)
        self.assertEqual(self.se.name_to_duration("20133.333333333332--21133.333333333332"), 1000.0)
        self.assertEqual(self.se.name_to_duration("43533.33333333333--44200.0"), 666.6666666666715)

    def test_run_extraction(self):
        print("\nRunning test for run_extraction method...")
        sub_area = (288, 958, 1632, 1044)
        test_sub_path = self.se.run_extraction(ch_vid, sub_area)
        test_sub_txt = test_sub_path.read_text(encoding="utf-8")
        test_sub_path.unlink()
        self.assertEqual(test_sub_txt, ch_vid_srt.read_text(encoding="utf-8"))
