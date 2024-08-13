
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
        print(f"Executing: {app_path} {args}")

        result = subprocess.run([app_path] + args.split(),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)

        if result.returncode != 0:
            print(f"Execution error: {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}")

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
            self.assertEqual(len(chksums), 304)
            self.assertTrue(os.path.isfile(input_file))

            pure_run_chksums = set(self.all_chksums.values())
            base_run_chksums = set(chksums.values())
            self.assertEqual(pure_run_chksums, base_run_chksums)


def main(app_path):
    TestAstroStacker.AS_PATH = app_path
    unittest.main()


if __name__ == "__main__":
    as_path = environ.get('AS_PATH')
    main(as_path)
