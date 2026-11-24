import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from flymerge.game import observe
from flymerge.robust_planner import choose
from flymerge.spawn_contact import (SITE_RADII, SPAWN_OFFSET_PX, SPAWN_Y, TEST_Y, carry_hidden, check, gate,
                                    label_actions, target_screen_x)

FIXTURES = Path(__file__).parent/'fixtures'/'spawn'
CACHE = Path('data/site')


class ContactRule(unittest.TestCase):
    def test_calibrated_lines(self):
        self.assertAlmostEqual(SPAWN_Y, -38*440/565, places=6)
        self.assertLess(TEST_Y, SPAWN_Y)

    def test_orange_onto_high_apple_is_fatal(self):
        # Apple top just under the spawn: an orange released straight above lands within the test band.
        apple = (200., SPAWN_Y+SITE_RADII[4]+SITE_RADII[5]-4, 5)
        self.assertEqual(check([apple], 4, 200.)['label'], 'fatal')

    def test_same_tier_high_contact_is_uncertain(self):
        orange = (200., SPAWN_Y+2*SITE_RADII[4]-4, 4)
        self.assertEqual(check([orange], 4, 200.)['label'], 'uncertain')

    def test_low_pile_is_clear(self):
        self.assertEqual(check([(200., 300., 5)], 4, 200.)['label'], 'clear')
        self.assertEqual(check([], 2, 200.)['label'], 'clear')

    def test_borderline_is_uncertain(self):
        # Contact height that just fails the test for the held fruit lands in the borderline band.
        r = SITE_RADII[4]
        edge = TEST_Y+r/2
        fruit = (100., edge+SITE_RADII[3]+r+2, 3)
        self.assertEqual(check([fruit], 4, 100.)['label'], 'uncertain')

    def test_cherry_resting_at_spawn_is_not_certain(self):
        under = (100., SPAWN_Y+SITE_RADII[0]+SITE_RADII[2]-10, 2)
        self.assertEqual(check([under], 0, 100.)['label'], 'uncertain')

    def test_gate_requires_an_alternative(self):
        fatal = dict(label='fatal')
        self.assertEqual(gate([fatal]*15, 3)[0], False)
        labels = [fatal]*14+[dict(label='uncertain')]
        block, allowed = gate(labels, 3)
        self.assertTrue(block)
        self.assertEqual(list(np.flatnonzero(allowed)), [14])

    def test_strict_gate_prefers_clear(self):
        labels = [dict(label='uncertain')]*3+[dict(label='clear')]*2+[dict(label='fatal')]*10
        self.assertEqual(gate(labels, 0)[0], False)
        block, allowed = gate(labels, 0, strict=True)
        self.assertTrue(block)
        self.assertEqual(list(np.flatnonzero(allowed)), [3, 4])
        # Without a clear aim, strict falls back to the non-fatal rule.
        only = [dict(label='uncertain')]*2+[dict(label='fatal')]*13
        self.assertEqual(gate(only, 0, strict=True)[0], False)
        self.assertEqual(gate(only, 5, strict=True)[0], True)

    def test_mask_memory_keeps_hidden_corner_grape(self):
        # Website attempt 3, drops 232 -> 233: the grape at the right rim vanished under the orange's mask.
        box = (370, 216, 896, 781)
        grape = (386.6, 12.5, 2)
        hand = (827., box[1]+SPAWN_OFFSET_PX, 3)
        self.assertEqual(carry_hidden([grape], [], hand, box), [grape])
        # Seen again, or outside the mask column, or low in the pile: not carried.
        self.assertEqual(carry_hidden([grape], [(388., 14., 2)], hand, box), [])
        self.assertEqual(carry_hidden([grape], [], (500., hand[1], 3), box), [])
        self.assertEqual(carry_hidden([(386.6, 300., 2)], [], hand, box), [])
        self.assertEqual(carry_hidden([grape], [], None, box), [])

    def test_target_matches_adapter(self):
        x0, x1 = 370, 896
        self.assertAlmostEqual(target_screen_x(14, 1, x0, x1), x1-(63*.34+3))
        self.assertEqual(len(label_actions([], 1, (x0, 216, x1, 781))), 15)


class RobustMask(unittest.TestCase):
    def test_unmasked_matches_legacy_and_mask_excludes(self):
        obs = observe([(.2, .9, 3), (.8, .85, 2), (.5, .95, 4)], 1, 2)
        logp = np.log(np.full(15, 1/15))
        a, trace = choose(obs, logp, np.random.default_rng(5))
        b, _ = choose(obs, logp, np.random.default_rng(5), allowed=np.ones(15, bool))
        self.assertEqual(a, b)
        allowed = np.ones(15, bool)
        allowed[a] = False
        rng = np.random.default_rng(5)
        c, masked = choose(obs, logp, rng, allowed=allowed)
        self.assertNotEqual(c, a)
        self.assertNotIn(a, masked['candidates'])
        # Same random draws are consumed with or without the mask.
        other = np.random.default_rng(5)
        choose(obs, logp, other)
        self.assertEqual(rng.bit_generator.state, other.bit_generator.state)


@unittest.skipUnless((CACHE/'10.png').exists(), 'site sprites not cached')
class ExtendedDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flymerge.site import Vision
        cls.meta = json.loads((FIXTURES/'fixtures.json').read_text())
        cls.legacy = Vision(CACHE, 4)
        cls.improved = Vision(CACHE, 4, watermelon_inner=True)

    def frame(self, run, name):
        x0, y0, x1, y1 = self.meta['crop']
        full = np.zeros((900, 1280, 3), np.uint8)
        full[y0:y1, x0:x1] = np.asarray(Image.open(FIXTURES/f'{run}-{name}.png').convert('RGB'))
        return full

    def detect(self, run, name):
        from flymerge.site import board_extended, hand_pixels, hand_position
        box = tuple(self.meta['runs'][run]['box'])
        x0, y0, x1, y1 = box
        pixels = self.frame(run, name)
        band = np.asarray(Image.open(FIXTURES/f'{run}-band-background.png').convert('RGB'))
        board = np.asarray(Image.open(FIXTURES/f'{run}-board-background.png').convert('RGB'))
        image, _ = hand_pixels(pixels, box)
        hand = hand_position(self.improved.match(image, range(5), (.64, .68, .72), (0,), .047), box)
        new, cov_new = board_extended(self.improved, pixels, box, band, hand, fast=True)
        old, cov_old = self.legacy.board(pixels[y0:y1, x0:x1], board, fast=True)
        screen = lambda found: [(x0+x*(x1-x0), y0+y*(y1-y0), t) for x, y, t, _ in found]
        return screen(old), cov_old, screen(new), cov_new, hand

    def assertFound(self, found, x, y, tier, tolerance=8):
        self.assertTrue(any(t == tier and abs(a-x) < tolerance and abs(b-y) < tolerance for a, b, t in found), found)

    def assertMissing(self, found, x, y, tier, tolerance=8):
        self.assertFalse(any(t == tier and abs(a-x) < tolerance and abs(b-y) < tolerance for a, b, t in found))

    def test_attempt1_corner_grape(self):
        old, _, new, _, hand = self.detect('attempt-1', '199-02062-settle')
        self.assertMissing(old, 864, 232, 2)
        self.assertFound(new, 864, 232, 2)
        # The held fruit is never reported as pile.
        self.assertMissing(new, hand[0], hand[1], hand[2], 20)

    def test_attempt2_corner_grape(self):
        old, _, new, _, hand = self.detect('attempt-2', '221-02330-settle')
        self.assertMissing(old, 866, 246, 2)
        self.assertFound(new, 866, 246, 2)
        self.assertMissing(new, hand[0], hand[1], hand[2], 20)

    def test_attempt3_watermelon(self):
        old, cov_old, new, cov_new, _ = self.detect('attempt-3', '182-01887-settle')
        self.assertLess(cov_old, .86)
        self.assertFalse(any(t == 10 for *_, t in old))
        self.assertGreaterEqual(cov_new, .86)
        self.assertFound(new, 668, 600, 10, 12)

    def test_ordinary_boards_unchanged(self):
        for run, name in (('attempt-1', '030-00331-settle'), ('attempt-3', '120-01240-settle')):
            old, _, new, _, _ = self.detect(run, name)
            self.assertEqual(sorted(t for *_, t in old), sorted(t for *_, t in new))
            self.assertFalse([f for f in new if f[1] < self.meta['runs'][run]['box'][1]])


if __name__ == '__main__':
    unittest.main()
