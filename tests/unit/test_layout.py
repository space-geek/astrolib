""" Test module which verifies synchronization of unit tests
    with source, with regards to both existance and layout.
"""

from pathlib import Path
import unittest

import astrolib

_SOURCE_PATH: Path = Path(astrolib.__file__).parent


class Test_UnitTestDirectoryLayout(unittest.TestCase):
    """Test case for unit test directory layout."""

    def setUp(self) -> None:
        # Construct a handle to this file:
        self.this_file_path: Path = Path(__file__)

    def test_this_file_is_as_expected(self):
        """Test for verifying this file is in the expected location for all relative
        path construction in this module.
        """
        self.assertTrue(self.this_file_path.parent.name == "unit")
        self.assertTrue(self.this_file_path.parent.parent.name == "tests")

    @unittest.expectedFailure  # TODO Add unit test modules for all modules
    def test_module_per_src_module(self):
        """Test for verifying unit test modules exist for each source module."""

        # Iterate over the source modules:
        for source_file_path in _SOURCE_PATH.glob("**/*.py"):

            # Ignore __init__.py file(s):
            if source_file_path.name == "__init__.py":
                continue

            # Construct the expected test module path:
            test_relative_path: Path = source_file_path.relative_to(_SOURCE_PATH).parent
            test_filename: str = f"test_{source_file_path.stem}.py"
            test_path: Path = Path(
                self.this_file_path.parent,
                test_relative_path,
                test_filename,
            )

            # Validate test file existance:
            error_msg: str = f"{source_file_path} found, but {test_path} is missing."
            self.assertTrue(test_path.is_file(), msg=error_msg)

    def test_unit_matches_src(self):
        """Test for verifying unit test file layout is synchronized with source file
        layout.
        """

        # Cache the parent directory for this file:
        tests_dir: Path = self.this_file_path.parent

        # Iterate over the unit test modules:
        for test_file_path in tests_dir.glob("**/test_*.py"):
            # Skip this file:
            if test_file_path == self.this_file_path:
                continue

            # Construct the expected source module path:
            source_relative_path: Path = test_file_path.relative_to(tests_dir).parent
            source_filename: str = test_file_path.name.split("test_", maxsplit=1)[1]
            source_path: Path = Path(
                _SOURCE_PATH,
                source_relative_path,
                source_filename,
            )

            # Validate source file existance:
            error_msg: str = f"{test_file_path} found, but {source_path} is missing."
            self.assertTrue(source_path.is_file(), msg=error_msg)
