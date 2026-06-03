import base64
import unittest

from fastapi import HTTPException

from helper import check_it_decode
from items import APIRequest


def _req(transaction: str) -> APIRequest:
    return APIRequest(header="h", transaction=transaction)


# A tiny but real payload so we can assert exact round-trip bytes.
PAYLOAD = b"\x89PNG\r\n\x1a\n-hello-\xff\xd8\xff"


class CheckItDecodeTests(unittest.TestCase):
    def test_plain_base64(self):
        encoded = base64.b64encode(PAYLOAD).decode()
        self.assertEqual(check_it_decode(_req(encoded)), PAYLOAD)

    def test_data_uri_prefix_is_stripped(self):
        encoded = base64.b64encode(PAYLOAD).decode()
        uri = f"data:image/png;base64,{encoded}"
        self.assertEqual(check_it_decode(_req(uri)), PAYLOAD)

    def test_data_uri_with_charset_params(self):
        encoded = base64.b64encode(PAYLOAD).decode()
        uri = f"data:image/jpeg;charset=utf-8;base64,{encoded}"
        self.assertEqual(check_it_decode(_req(uri)), PAYLOAD)

    def test_url_safe_alphabet(self):
        # Choose bytes whose standard encoding contains + and / so the
        # url-safe variant actually differs (-/_).
        payload = b"\xfb\xff\xbf\xfe\xff"
        encoded = base64.urlsafe_b64encode(payload).decode()
        self.assertIn("-", encoded + base64.urlsafe_b64encode(b"\xfb\xef").decode())
        self.assertEqual(check_it_decode(_req(encoded)), payload)

    def test_missing_padding_is_restored(self):
        encoded = base64.b64encode(PAYLOAD).decode().rstrip("=")
        self.assertEqual(check_it_decode(_req(encoded)), PAYLOAD)

    def test_whitespace_and_newlines_are_ignored(self):
        encoded = base64.b64encode(PAYLOAD).decode()
        chunked = "\n".join(encoded[i : i + 4] for i in range(0, len(encoded), 4))
        wrapped = f"  {chunked}  \n"
        self.assertEqual(check_it_decode(_req(wrapped)), PAYLOAD)

    def test_invalid_base64_raises_400(self):
        with self.assertRaises(HTTPException) as ctx:
            check_it_decode(_req("not valid base64 @@@@ !!!!"))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_empty_transaction_raises_400(self):
        with self.assertRaises(HTTPException) as ctx:
            check_it_decode(_req(""))
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
