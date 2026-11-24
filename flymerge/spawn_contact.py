"""Candidate check for the site's early high-contact game-over rule.

The site's collision handler (public script, read as text) ends the game when two fruits of different
tiers touch and either body satisfies position.y - centerOffset.y/2 < 70 game px; for a circle body
the offset is its radius. Same-tier contacts merge instead (the new fruit appears at one body's
position), so they are not fatal by this branch but can be when high, and are marked uncertain.

Geometry is in local transfer units (board box 410 x 440, box top y = 0) using the browser's
measured box (526 x 565 screen px). The held fruit's center was measured at screen y = box top - 38
on all 60 sampled aim frames, which is game y = 80. One game px is 0.963 screen px (canvas 724 game px
tall drawn 697 px tall), so the site's test line for a body is center - r/2 < 70 game px, i.e. 10 game
px above the spawn center.
"""
import numpy as np

from .game import AIMS

BOX_W, BOX_H = 526, 565
LOCAL_W, LOCAL_H = 410, 440
SCREEN_PER_GAME = .963
SPAWN_OFFSET_PX = -38.                      # held-fruit center relative to the box top, screen px
TO_LOCAL = LOCAL_H/BOX_H
SPAWN_Y = SPAWN_OFFSET_PX*TO_LOCAL          # held-fruit center, local units
TEST_Y = (SPAWN_OFFSET_PX-10*SCREEN_PER_GAME)*TO_LOCAL   # center - r/2 must stay at or below this
# Collision radii from the site's sprites, setSize rules and setScale(0.75), local units.
SITE_RADII = np.array([13.8, 17.7, 24.8, 27.0, 34.3, 43.9, 51.2, 62.8, 70.7, 87.0, 101.3])
SPRITE_WIDTHS = (49, 63, 88, 96, 126, 164, 184, 223, 251, 309, 360)
MARGIN = 6.          # detector center uncertainty applied to contact height, local units (~7.7 screen px)
TOL = 1.             # calibration tolerance of the test line itself, local units (~1.3 screen px)
GRAZE = 6.           # horizontal reach for high neighbours passed on the way down


def target_screen_x(action, tier, x0, x1):
    """The browser adapter's release x for an action (same formula as flymerge.site)."""
    margin = SPRITE_WIDTHS[tier]*.34+3
    return x0+margin+(x1-x0-2*margin)*AIMS[action]


def check(fruits, held, x):
    """Label one release at local x for held tier `held` against fruits [(x, y, tier)].

    The first contact height is uncertain by +-MARGIN (detected centers); it cannot be above the spawn
    center. 'fatal' means the site's test fires even at the least favourable contact height for a
    different-tier contact; 'uncertain' means it fires for some height in the band, or the contact is
    same-tier and high; 'clear' otherwise. Fruits may have negative y (above the box top).
    """
    r = SITE_RADII[held]
    first = None
    for fx, fy, tier in fruits:
        reach = r+SITE_RADII[int(tier)]
        dx = abs(fx-x)
        if dx >= reach:
            continue
        raw = fy-np.sqrt(reach*reach-dx*dx)
        if first is None or raw < first[0]:
            first = (raw, fx, fy, int(tier))
    result = dict(label='clear', slack=None, contact=None)
    if first is None:
        return result
    raw, fx, fy, tier = first
    own_best = max(SPAWN_Y, raw-MARGIN)-r/2-TEST_Y
    own_worst = max(SPAWN_Y, raw+MARGIN)-r/2-TEST_Y
    other = fy-SITE_RADII[tier]/2-TEST_Y
    surely = own_worst < -TOL or other < -MARGIN
    possibly = own_best < TOL or other < MARGIN
    result.update(slack=round(float(min(own_worst, other)), 2),
                  contact=dict(center_y=round(float(max(SPAWN_Y, raw)), 2), x=round(float(fx), 1),
                               y=round(float(fy), 1), tier=tier+1, same_tier=tier == held))
    if tier == held:
        result['label'] = 'uncertain' if possibly else 'clear'
    elif surely:
        result['label'] = 'fatal'
    elif possibly:
        result['label'] = 'uncertain'
    if result['label'] == 'clear':
        # A high different-tier neighbour beside the path can be touched while sliding.
        for fx, fy, t in fruits:
            if int(t) != held and abs(fx-x) < r+SITE_RADII[int(t)]+GRAZE and fy-SITE_RADII[int(t)]/2-TEST_Y < MARGIN:
                result['label'] = 'uncertain'
                result['graze'] = dict(x=round(float(fx), 1), y=round(float(fy), 1), tier=int(t)+1)
                break
    return result


def label_actions(fruits, held, box):
    """Contact labels for all 15 actions, given fruits in local units and the screen board box."""
    x0, _, x1, _ = box
    return [check(fruits, held, (target_screen_x(a, held, x0, x1)-x0)*LOCAL_W/(x1-x0)) for a in range(15)]


def gate(labels, action, strict=False):
    """Block `action` only when it is high-confidence fatal and some action is not fatal.

    With `strict`, an uncertain action is also blocked when some action is clear (every terminal drop
    in the first six website games that the check labelled uncertain had clear alternatives)."""
    names = [l['label'] for l in labels]
    if strict and 'clear' in names:
        allowed = np.array([n == 'clear' for n in names])
    else:
        allowed = np.array([n != 'fatal' for n in names])
    block = not allowed[action] and allowed.any()
    return bool(block), allowed


def hand_mask(hand, box):
    """Screen-px zone that `site.board_extended` excludes around the held fruit: (left, right, bottom)."""
    hx, hy, tier = hand
    radius = SITE_RADII[tier]*BOX_H/LOCAL_H
    half = max(58, radius+10)
    return hx-half, hx+half, hy+radius+4


def carry_hidden(previous, current, hand, box, reach=12.):
    """Fruit from the previous board that the hand mask now hides, to keep in the contact check.

    `previous` and `current` are [(x, y, tier)] in local units; `hand` is the held fruit (screen x, y,
    tier) at detection time. A previous fruit is carried when its centre column lies under the mask,
    its top reaches into the masked rows, and no current fruit is within `reach` local units of it.
    Website attempt 3 (seed 50003) ended when a corner grape seen on drop 232 was hidden this way on
    drop 233 and the held orange was aimed onto it.
    """
    if hand is None:
        return []
    x0, y0, x1, y1 = box
    left, right, bottom = hand_mask(hand, box)
    sx = lambda x: x0+x*(x1-x0)/LOCAL_W
    sy = lambda y: y0+y*(y1-y0)/LOCAL_H
    carried = []
    for x, y, t in previous:
        top = sy(y-SITE_RADII[t])
        if not (left <= sx(x) <= right and top < bottom):
            continue
        if any(np.hypot(x-cx, y-cy) < max(reach, .5*SITE_RADII[t]) for cx, cy, _ in current):
            continue
        carried.append((x, y, t))
    return carried
