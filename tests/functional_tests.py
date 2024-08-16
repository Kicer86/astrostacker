
import hashlib
import subprocess
import unittest
import tempfile
import os
from os import sys, environ


def calculate_checksums(directory):
    checksums = {}

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
                checksums[file_path] = file_hash.hexdigest()

    return checksums


def filter_checksums(checksums, filter):
    filtered_checksums = {}

    for file, chksum in checksums.items():
        if not any(substring in file for substring in filter):
            filtered_checksums[file] = chksum

    return filtered_checksums


def run_application(app_path, args=""):
    """
    Runs the console application with the specified arguments and returns the output.

    Parameters:
        app_path (str): The path to the console application.
        args (str): Command line arguments to pass to the application.

    Returns:
        str: The output of the console application.
    """
    try:
        result = subprocess.run([app_path] + args.split(),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)

        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return None, str(e), -1


class TestAstroStacker(unittest.TestCase):
    AS_PATH = ""

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = "video-files/moon.mp4"
            stdout, stderr, code = run_application(cls.AS_PATH, f"--working-dir {temp_dir} {input_file}")
            cls.all_chksums = calculate_checksums(temp_dir)

    def test_base_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = "video-files/moon.mp4"
            stdout, stderr, code = run_application(self.AS_PATH, f"--working-dir {temp_dir} {input_file}")
            self.assertEqual(code, 0);

            chksums = calculate_checksums(temp_dir)
            self.assertEqual(len(chksums), 244)
            self.assertTrue(os.path.isfile(input_file))

            pure_run_chksums = set(self.all_chksums.values())
            base_run_chksums = set(chksums.values())
            self.assertEqual(pure_run_chksums, base_run_chksums)

    def test_split_option(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = "video-files/moon.mp4"
            stdout, stderr, code = run_application(self.AS_PATH, f"--working-dir {temp_dir} --split 15,0 {input_file}")
            self.assertEqual(code, 0);

            chksums = calculate_checksums(temp_dir)
            self.assertEqual(len(chksums), 260)
            self.assertTrue(os.path.isfile(input_file))

            # compare results but remove elements which will be different
            pure_run_chksums = filter_checksums(self.all_chksums, ["aligned", "enhanced", "stacked"])
            split_run_chksums = filter_checksums(chksums, ["aligned", "enhanced", "stacked"])
            self.assertEqual(set(pure_run_chksums.values()), set(split_run_chksums.values()))

    def test_noop_crop_option(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = "video-files/moon.mp4"
            stdout, stderr, code = run_application(self.AS_PATH, f"--working-dir {temp_dir} --crop 10000x10000,0,0 {input_file}")   # width and heigh are bigger than actual image size so there should be no difference
            self.assertEqual(code, 0);

            chksums = calculate_checksums(temp_dir)
            self.assertEqual(len(chksums), 304)
            self.assertTrue(os.path.isfile(input_file))

            # cropped set has more images (crop step) but they should be duplicates of images from previous set, so set() should remove them as duplicates
            pure_run_chksums = set(self.all_chksums.values())
            base_run_chksums = set(chksums.values())
            self.assertEqual(pure_run_chksums, base_run_chksums)

    def test_crop_option(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = "video-files/moon.mp4"
            stdout, stderr, code = run_application(self.AS_PATH, f"--working-dir {temp_dir} --crop 100x200,-50,70 {input_file}")
            self.assertEqual(code, 0);

            chksums = calculate_checksums(temp_dir)
            self.assertEqual(len(chksums), 304)
            self.assertTrue(os.path.isfile(input_file))

            pure_run_chksums = set(self.all_chksums.values())
            base_run_chksums = set(chksums.values())
            self.assertNotEqual(pure_run_chksums, base_run_chksums)

def main(app_path):
    TestAstroStacker.AS_PATH = app_path
    unittest.main()


if __name__ == "__main__":
    as_path = environ.get('AS_PATH')
    main(as_path)
