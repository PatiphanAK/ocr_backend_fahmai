"""Decode tests driven by real records sampled from ocr_extracted.json.

Background: the production `UnidentifiedImageError` traces back to the input
data, not the decoder. In ocr_extracted.json, 536/752 records are macOS
AppleDouble sidecar metadata (the `._filename` xattr blobs, magic 00 05 16 07)
rather than the actual document. Those broken records all share one signature:
`header == transaction`. The remaining 216 records are genuine PNG scans.

`test_fixtures.json` holds a small real sample of each kind so these tests stay
self-contained instead of loading the full 72 MB file.
"""

import base64
import json
import os
import unittest

from fastapi import HTTPException

from helper import check_it_decode
from items import APIRequest

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "test_fixtures.json")

PNG_MAGIC = bytes.fromhex("89504e47")
APPLEDOUBLE_MAGIC = bytes.fromhex("00051607")


def _load_fixture():
    with open(FIXTURE_PATH) as fh:
        return json.load(fh)


def _req(record):
    return APIRequest(header=record["header"], transaction=record["transaction"])


class DecodeRealImageRecords(unittest.TestCase):
    """The 216 good records: valid base64 that decodes to a real PNG."""

    @classmethod
    def setUpClass(cls):
        cls.records = _load_fixture()["images"]

    def test_fixture_has_image_records(self):
        self.assertGreater(len(self.records), 0)

    def test_decode_returns_png_bytes(self):
        for i, rec in enumerate(self.records):
            with self.subTest(record=i):
                document = check_it_decode(_req(rec))
                self.assertTrue(document.startswith(PNG_MAGIC))
                # check_it_decode must not alter the payload.
                self.assertEqual(document, base64.b64decode(rec["transaction"]))

    def test_good_records_differ_from_header(self):
        # Real document records do NOT have header == transaction.
        for i, rec in enumerate(self.records):
            with self.subTest(record=i):
                self.assertNotEqual(rec["header"], rec["transaction"])


class DecodeMetadataSidecarRecords(unittest.TestCase):
    """The 536 broken records: valid base64, but AppleDouble metadata."""

    @classmethod
    def setUpClass(cls):
        cls.records = _load_fixture()["metadata_sidecars"]

    def test_fixture_has_metadata_records(self):
        self.assertGreater(len(self.records), 0)

    def test_decode_succeeds_so_bug_is_not_in_decoder(self):
        # The base64 itself is valid; check_it_decode returns bytes without
        # raising. The data is just not an image — proving the failure is
        # upstream (bad input), not in the decode step.
        for i, rec in enumerate(self.records):
            with self.subTest(record=i):
                document = check_it_decode(_req(rec))
                self.assertTrue(document.startswith(APPLEDOUBLE_MAGIC))

    def test_broken_records_have_header_equal_transaction(self):
        # This is the tell-tale signature of the corrupt records.
        for i, rec in enumerate(self.records):
            with self.subTest(record=i):
                self.assertEqual(rec["header"], rec["transaction"])


try:
    from pipeline import load_document

    _HAS_PIPELINE = True
except Exception:  # noqa: BLE001 - heavy ML deps may be unavailable
    _HAS_PIPELINE = False


@unittest.skipUnless(_HAS_PIPELINE, "pipeline (and its ML deps) not importable")
class LoadDocumentOnRealData(unittest.TestCase):
    """End-to-end of the decode path: bytes -> pages, per record kind."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = _load_fixture()

    def test_image_records_produce_one_rgb_page(self):
        for i, rec in enumerate(self.fixture["images"]):
            with self.subTest(record=i):
                pages = load_document(check_it_decode(_req(rec)))
                self.assertEqual(len(pages), 1)
                self.assertEqual(pages[0]["image"].mode, "RGB")

    def test_metadata_records_raise_clear_value_error(self):
        for i, rec in enumerate(self.fixture["metadata_sidecars"]):
            with self.subTest(record=i):
                document = check_it_decode(_req(rec))
                with self.assertRaises(ValueError) as ctx:
                    load_document(document)
                self.assertIn("not a readable image or PDF", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
