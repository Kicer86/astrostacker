
import subprocess
import unittest
import tempfile
from os import sys, environ


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

    def test_base_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout, stderr, code = run_application(self.AS_PATH, f"--working-dir {temp_dir} video-files/moon.mp4")
            self.assertEqual(code, 0);


def main(app_path):
    TestAstroStacker.AS_PATH = app_path
    unittest.main()


if __name__ == "__main__":
    as_path = environ.get('AS_PATH')
    main(as_path)
