import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from flymerge.site import filter_ad_requests


class RequestFilterTest(unittest.TestCase):
    def test_ad_domains_and_site_resources(self):
        for url, blocked in [
            ('https://googleads.g.doubleclick.net/pagead/iframe', True),
            ('https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js', True),
            ('https://playsuikagame.com/static/assets/sprite/2.png', False),
            ('https://unrelated-doubleclick.net/sprite.png', False),
        ]:
            route = SimpleNamespace(request=SimpleNamespace(url=url), abort=Mock(), continue_=Mock())
            filter_ad_requests(route)
            self.assertEqual(route.abort.called, blocked)
            self.assertEqual(route.continue_.called, not blocked)
