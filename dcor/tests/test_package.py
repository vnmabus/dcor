"""Tests of the dcor package."""

import re
import unittest

import dcor


class TestVersion(unittest.TestCase):
    """Tests of the version number."""

    def test_version(self) -> None:
        """Test that the version has the right format."""
        regex = re.compile(r"\d+\.\d+(\.\d+)?")
        self.assertTrue(regex.match(dcor.__version__))
        self.assertNotEqual(dcor.__version__, "0.0")
