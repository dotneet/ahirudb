#!/usr/bin/env python3
"""Regression tests for the differential probe's output normalizer."""

import unittest

from diff_probe import NULL_TOKEN, normalize


class NormalizeTests(unittest.TestCase):
    def test_empty_string_and_null_are_distinct(self):
        self.assertEqual(
            normalize(f"a,b,c,d\n,{NULL_TOKEN},NULL,\n"),
            [("", "<null>", "NULL", "")],
        )

    def test_trailing_empty_columns_are_kept(self):
        self.assertEqual(normalize("a,b,c\nx,,\n"), [("x", "", "")])

    def test_csv_quotes_and_embedded_newline_are_kept(self):
        self.assertEqual(
            normalize(f'a,b\n"line\nbreak",{NULL_TOKEN}\n'),
            [("line\nbreak", "<null>")],
        )

    def test_string_whitespace_is_not_stripped(self):
        self.assertEqual(normalize("a,b\n  x  , y\n"), [("  x  ", " y")])


if __name__ == "__main__":
    unittest.main()
