"""Run one headed Playwright episode using screenshot-only fruit observations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import time
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urlsplit

from .policy import np, restore
from .game import AIMS, observe
from .replay import json_file, pack_script
import cv2
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

URL = 'https://playsuikagame.com/ko'
cv2.setNumThreads(1)
RECORDER = None
AD_HOSTS = ('doubleclick.net', 'googlesyndication.com', 'googleadservices.com',
            'adtrafficquality.google', 'adnxs.com', 'adsrvr.org')


def filter_ad_requests(route):
    host = urlsplit(route.request.url).hostname or ''
    if any(host == domain or host.endswith('.' + domain) for domain in AD_HOSTS):
        route.abort()
    else:
        route.continue_()


class Vision:
    def __init__(self, cache, workers=8, watermelon_inner=False):
        cache.mkdir(parents=True, exist_ok=True)
        # Tier 11 is matched on its inner 65% disk: its edge is usually covered by neighbours.
        self.watermelon_inner = watermelon_inner
        self.sprites = []
        self.templates = {}
        self.pool = ThreadPoolExecutor(max_workers=workers)
        for tier in range(11):
            path = cache / f'{tier}.png'
            if not path.exists():
                urlretrieve(f'https://playsuikagame.com/static/assets/sprite/{tier}.png', path)
            self.sprites.append(np.asarray(Image.open(path).convert('RGBA')))

    def template(self, tier, scale, angle):
        key = (tier, scale, angle)
        if key in self.templates:
            return self.templates[key]
        rgba = self.sprites[tier]
        size = tuple(max(1, round(x*scale)) for x in rgba.shape[1::-1])
        rgba = cv2.resize(rgba, size, interpolation=cv2.INTER_AREA)
        radius = max(size) * .62
        if angle:
            side = int(np.ceil(np.hypot(*size)))
            matrix = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1)
            matrix[:, 2] += (side/2-size[0]/2, side/2-size[1]/2)
            rgba = cv2.warpAffine(rgba, matrix, (side, side))
        mask = (rgba[:, :, 3] > 230).astype(np.uint8)
        if tier == 10 and self.watermelon_inner:
            h, w = mask.shape
            yy, xx = np.mgrid[:h, :w]
            mask &= ((xx-w/2)**2+(yy-h/2)**2 < (.65*radius/.62/2)**2).astype(np.uint8)
        item = np.ascontiguousarray(rgba[:, :, :3]), mask, radius
        self.templates[key] = item
        return item

    def match(self, pixels, tiers, scales, angles, threshold, multi=False, changed=None, candidates_only=False):
        if multi and not candidates_only:
            jobs = [self.pool.submit(self.match, pixels, (tier,), scales, angles, threshold,
                                     True, changed, True) for tier in tiers]
            matches = sorted(candidate for job in jobs for candidate in job.result())
            return self.suppress(matches)
        matches = []
        if multi:
            ys, xs = np.where(changed)
            if not len(xs):
                return []
            xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
        for tier in tiers:
            for scale in scales:
                for angle in angles:
                    pattern, mask, radius = self.template(tier, scale, angle)
                    h, w = mask.shape
                    if h > pixels.shape[0] or w > pixels.shape[1]:
                        continue
                    dx = dy = 0
                    subject = pixels
                    if multi:
                        dx, dy = max(0, xmin-w//2), max(0, ymin-h//2)
                        right, bottom = xmax+w-w//2, ymax+h-h//2
                        subject = pixels[dy:bottom, dx:right]
                        if h > subject.shape[0] or w > subject.shape[1]:
                            continue
                    result = cv2.matchTemplate(subject, pattern, cv2.TM_SQDIFF_NORMED, mask=mask)
                    result[~np.isfinite(result)] = 1
                    if multi:
                        minima = (result < threshold) & (result == cv2.erode(result, None))
                        ys, xs = np.where(minima)
                        for x, y in zip(xs, ys):
                            cx, cy = x+dx+w//2, y+dy+h//2
                            if changed[cy, cx]:
                                matches.append((float(result[y,x]), cx, cy, tier, radius))
                    else:
                        value, _, (x, y), _ = cv2.minMaxLoc(result)
                        if value < threshold:
                            matches.append((value, x+w//2, y+h//2, tier, radius))
        matches.sort()
        if not multi:
            return matches[0] if matches else None
        return matches

    @staticmethod
    def suppress(matches):
        retained = []
        for candidate in matches:
            _, x, y, _, r = candidate
            if all(np.hypot(x-a, y-b) > .70*(r+s) for _, a, b, _, s in retained):
                retained.append(candidate)
        return retained

    def board(self, pixels, background, fast=False):
        pixels = cv2.resize(pixels, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        background = cv2.resize(background, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        changed = np.max(cv2.absdiff(pixels, background), axis=2) > 43
        # The held fruit can overlap the top few pixels of the box.
        changed[:6] = False
        if not changed.any():
            return [], 1.
        ys, xs = np.where(changed)
        x0, y0 = max(0, xs.min()-92), max(0, ys.min()-92)
        x1, y1 = min(pixels.shape[1], xs.max()+93), min(pixels.shape[0], ys.max()+93)
        pad = 80
        search = cv2.copyMakeBorder(pixels, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        candidates = np.pad(changed, pad)
        left, top = x0 + (pad if x0 else 0), y0 + (pad if y0 else 0)
        right = x1 + (pad if x1 < pixels.shape[1] else 2*pad)
        bottom = y1 + (pad if y1 < pixels.shape[0] else 2*pad)
        scales = (.35,) if fast else (.337, .365)
        angles = range(0, 360, 45 if fast else 30)
        detected = self.match(search[top:bottom, left:right], range(11), scales,
                              angles, .074, True, candidates[top:bottom, left:right])
        x0, y0 = left-pad, top-pad
        covered = np.zeros(changed.shape, np.uint8)
        result = []
        for quality, x, y, tier, radius in detected:
            x, y = int(x+x0), int(y+y0)
            cv2.circle(covered, (x,y), round(radius), 1, -1)
            result.append((x/pixels.shape[1], y/pixels.shape[0], tier, quality))
        return result, float(covered[changed].mean())


class Recorder:
    """Save frames the adapter already captures; JPEG encoding runs off the control thread."""

    def __init__(self, root, origin):
        self.root = Path(root)
        (self.root/'frames').mkdir(parents=True, exist_ok=True)
        self.origin = origin
        self.drop = 0
        self.index = []
        self.pool = ThreadPoolExecutor(max_workers=1)

    def add(self, tag, captured, image):
        t = captured-self.origin
        name = f'frames/{self.drop:03}-{len(self.index):05}-{tag}.jpg'
        self.index.append(dict(drop=self.drop, t=round(t, 4), tag=tag, image=name))
        self.pool.submit(lambda: Image.fromarray(image).save(self.root/name, quality=80))

    def close(self, extra):
        self.pool.shutdown(wait=True)
        json_file(self.root/'frames.json', dict(**extra, frames=self.index))


EXTENDED_UP, EXTENDED_SIDE = 110, 16


def extended_box(box):
    x0, y0, x1, y1 = box
    return x0-EXTENDED_SIDE, y0-EXTENDED_UP, x1+EXTENDED_SIDE, y1


RIM_ROWS = (4, 18)        # the box's front rim, drawn over fruit: screen rows y0+4 .. y0+18
RIM_ZONE = 40             # fruit centered up to this far below the box top get rim-aware matching


def masked_scores(image, valid, pattern, mask):
    """TM_SQDIFF_NORMED restricted to pixels where `valid` is 1, plus the visible template fraction."""
    m = mask.astype(np.float32)
    t = pattern.astype(np.float32)
    def corr(src, kernel):
        out = cv2.filter2D(src, -1, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
        return out[:src.shape[0]-kernel.shape[0]+1, :src.shape[1]-kernel.shape[1]+1]
    a = corr(valid, (m[..., None]*t*t).sum(2))
    b = sum(corr(image[..., c]*valid, m*t[..., c]) for c in range(3))
    c = corr((image*image).sum(2)*valid, m)
    seen = corr(valid, m)/max(1., m.sum())
    with np.errstate(all='ignore'):
        score = (a-2*b+c)/np.sqrt(a*c)
    score[~np.isfinite(score)] = 1
    return score, seen


def rim_matches(vision, region, changed, valid, rim_center_rows, tiers, scales, angles, threshold):
    """Template matches for fruit centered near the box top, ignoring the rim rows."""
    image = region.astype(np.float32)
    near = cv2.dilate(changed.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    found = []
    for tier in tiers:
        for scale in scales:
            for angle in angles:
                pattern, mask, radius = vision.template(tier, scale, angle)
                h, w = mask.shape
                rows = min(region.shape[0], rim_center_rows+h-h//2)
                if rows < h or w > region.shape[1]:
                    continue
                score, seen = masked_scores(image[:rows], valid[:rows], pattern, mask)
                minima = (score < threshold) & (seen >= .6) & (score == cv2.erode(score, None))
                for y, x in zip(*np.where(minima)):
                    cx, cy = x+w//2, y+h//2
                    if cy <= rim_center_rows and near[cy, cx]:
                        found.append((float(score[y, x]), cx, cy, tier, radius))
    return found


def board_extended(vision, pixels, box, background, hand, fast=False):
    """Detect fruit in the board, the spawn band above it, and the upper wall corners.

    `pixels` is the full screenshot and `background` the extended region of a fruit-free frame.
    `hand` is (x, y, tier) of the held fruit in screen px; its column (and the cloud above it) is
    excluded so the held fruit is never reported as pile. Returns (x, y, tier, quality) normalized to
    the original board box, so fruit above the box top have y < 0.
    """
    from .spawn_contact import SITE_RADII, BOX_H, LOCAL_H
    x0, y0, x1, y1 = box
    ex0, ey0, ex1, ey1 = extended_box(box)
    region = cv2.resize(pixels[ey0:ey1, ex0:ex1], None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    base = cv2.resize(background, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    changed = np.max(cv2.absdiff(region, base), axis=2) > 43
    masked_rows = mask_left = mask_right = 0
    if hand is not None:
        hx, hy, tier = hand
        radius = SITE_RADII[tier]*BOX_H/LOCAL_H
        half = max(58, radius+10)
        mask_left, mask_right = int((hx-half-ex0)/2), int(np.ceil((hx+half-ex0)/2))
        masked_rows = int(np.ceil((hy+radius+4-ey0)/2))
        changed[:masked_rows, max(0, mask_left):max(0, mask_right)] = False
    if not changed.any():
        return [], 1.
    ys, xs = np.where(changed)
    pad = 80
    bx0, by0 = max(0, xs.min()-92), max(0, ys.min()-92)
    bx1, by1 = min(region.shape[1], xs.max()+93), min(region.shape[0], ys.max()+93)
    search = cv2.copyMakeBorder(region, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    candidates = np.pad(changed, pad)
    left, top = bx0+(pad if bx0 else 0), by0+(pad if by0 else 0)
    right = bx1+(pad if bx1 < region.shape[1] else 2*pad)
    bottom = by1+(pad if by1 < region.shape[0] else 2*pad)
    scales = (.35,) if fast else (.337, .365)
    angles = range(0, 360, 45 if fast else 30)
    detected = vision.match(search[top:bottom, left:right], range(11), scales, angles, .074, True,
                            candidates[top:bottom, left:right])
    detected = [(q, x+left-pad, y+top-pad, t, r) for q, x, y, t, r in detected]
    valid = np.ones(changed.shape, np.float32)
    valid[(y0+RIM_ROWS[0]-ey0)//2:(y0+RIM_ROWS[1]-ey0)//2+1] = 0
    valid[:masked_rows, max(0, mask_left):max(0, mask_right)] = 0
    rim = rim_matches(vision, region, changed, valid, (y0+RIM_ZONE-ey0)//2, range(11), scales, angles,
                      .074, ) if changed[:(y0+RIM_ZONE-ey0)//2].any() else []
    detected = Vision.suppress(sorted(detected+rim))
    covered = np.zeros(changed.shape, np.uint8)
    result = []
    for quality, x, y, tier, radius in detected:
        x, y = int(x), int(y)
        cv2.circle(covered, (x, y), round(radius), 1, -1)
        sx, sy = ex0+2*x, ey0+2*y
        result.append(((sx-x0)/(x1-x0), (sy-y0)/(y1-y0), tier, quality))
    return result, float(covered[changed].mean())


def compose_background(empty, shots, box, clearance=75):
    """Extended background: the empty-board frame, with band rows replaced by a per-column median over
    frames whose held fruit (and cloud) is at least `clearance` px from that column.

    `empty` is a full screenshot of the empty board; `shots` are (hand_x, full screenshot) pairs.
    """
    ex0, ey0, ex1, ey1 = extended_box(box)
    base = empty[ey0:ey1, ex0:ex1].astype(np.float32)
    rows = box[1]-ey0+12
    stack = np.full((len(shots), rows, ex1-ex0, 3), np.nan, np.float32)
    columns = np.arange(ex0, ex1)
    for k, (hx, image) in enumerate(shots):
        keep = np.abs(columns-hx) >= clearance
        stack[k][:, keep] = image[ey0:ey0+rows, ex0:ex1][:, keep]
    with np.errstate(all='ignore'):
        band = np.nanmedian(stack, axis=0)
    missing = np.isnan(band)
    band[missing] = base[:rows][missing]
    base[:rows] = band
    return base.astype(np.uint8), int(missing[..., 0].all(0).sum())


def hand_position(hand, box):
    """Screen (x, y, tier) of a hand match from `ready`."""
    if not hand:
        return None
    x0, y0, _, _ = box
    return hand[1]+max(0, x0-60), hand[2]+y0-104, hand[3]


def screenshot(page):
    return np.asarray(Image.open(io.BytesIO(page.screenshot(timeout=4000))).convert('RGB'))


def integer_text(page, selector):
    text = page.locator(selector).inner_text(timeout=1500).strip().replace(',', '')
    return int(text) if text.isdecimal() else None


def hand_pixels(pixels, box):
    x0, y0, x1, _ = box
    left = max(0, x0-60)
    return pixels[y0-104:y0+12, left:x1+60], left


def settling_frame(page, window):
    start = time.monotonic()
    image = screenshot(page)
    if RECORDER is not None:
        RECORDER.add('settle', (start+time.monotonic())/2, image)
    if window is not None:
        window.append(((start+time.monotonic())/2,image))
        del window[:-3]
    return image


def ready(page, detect, box, deadline, window=None):
    previous = None
    last_find = None
    quiet = 0
    while time.monotonic() < deadline:
        if page.locator('#restart-game').is_visible():
            return None, None, 'game over'
        pixels = settling_frame(page,window)
        x0, y0, x1, y1 = box
        board = cv2.resize(pixels[y0+12:y1, x0:x1], (132, 142), interpolation=cv2.INTER_AREA)
        hand, _ = hand_pixels(pixels, box)
        last_find = detect.match(hand, range(5), (.64, .68, .72), (0,), .047)
        if previous is not None and last_find and np.mean(cv2.absdiff(board, previous) > 30) < .005:
            quiet += 1
            if quiet >= 2:
                return pixels, last_find, None
        else:
            quiet = 0
        previous = board
        page.wait_for_timeout(180)
    return None, last_find, ('board stability timed out' if last_find else 'visible current fruit timed out')


def recover_ready(page, vision, box, background, limit, window=None, detect=None):
    """Confirm a ready fruit and stable centers when sprite animation trips pixel gating."""
    x0, y0, x1, y1 = box
    last = None
    while time.monotonic() < limit:
        if page.locator('#restart-game').is_visible():
            return None, None, 'game over'
        pixels = settling_frame(page,window)
        hand_image, _ = hand_pixels(pixels, box)
        hand = vision.match(hand_image, range(5), (.64,.68,.72), (0,), .047)
        if hand:
            if detect is None:
                fruits, coverage = vision.board(pixels[y0:y1,x0:x1], background, fast=True)
            else:
                fruits, coverage = detect(pixels, hand)
            if coverage >= .86:
                if last and len(last) == len(fruits) and all(
                    min((np.hypot(a-x, b-y) for x,y,t,_ in last if t == tier), default=1.) < .025
                    for a,b,tier,_ in fruits
                ):
                    return pixels, hand, None
                last = fruits
        page.wait_for_timeout(250)
    return None, None, 'fruit centers or ready cue remained uncertain'


def activity_panel(page, graph, decision):
    points = graph['positions']
    coords = np.array([p if p else [0,0,0] for p in points])
    if not np.any(coords):
        theta = np.arange(len(points)) * 2.399963
        coords = np.stack([np.cos(theta), np.sin(theta), theta*0], axis=1)
    coords = (coords - coords.min(0)) / np.maximum(1, coords.max(0) - coords.min(0))
    activity = decision['steps'][-1]['delta']
    regional = {}
    for source, _, _, annotations in graph['edges']:
        for name, count in annotations.items():
            if name == 'UNASGD':
                continue
            total, weight = regional.get(name, (0., 0))
            regional[name] = total + abs(activity[source]) * count, weight + count
    top = sorted(((name, total / weight) for name, (total, weight) in regional.items()),
                 key=lambda pair: -pair[1])[:3]
    page.evaluate("""({points,activity,drop,action,top}) => {
      let panel = document.getElementById('orchard-circuit');
      if (!panel) {
        panel = document.createElement('aside');
        panel.id = 'orchard-circuit';
        panel.style.cssText = 'position:fixed;right:22px;bottom:18px;width:274px;padding:12px;background:#14292b;color:#eef1da;border:2px solid #809e91;border-radius:12px;z-index:2147483647;font:14px system-ui;pointer-events:none';
        panel.innerHTML='<div></div><small style="color:#a8c7b7"></small><canvas width="246" height="152" style="display:block;width:246px;height:152px"></canvas>';
        document.body.appendChild(panel);
      }
      panel.firstChild.textContent = `Simulated circuit activity · Drop ${drop} · Aim ${action+1}/15`;
      panel.children[1].textContent = 'Step 6 · ' + top.map(([name, value]) => name + ' ' + value.toFixed(3)).join(' / ');
      const ctx = panel.lastChild.getContext('2d');
      ctx.clearRect(0,0,246,152);
      points.forEach(([x,y],i) => {
        const color = activity[i] < 0 ? [87,172,245] : [248,170,105], v = Math.abs(activity[i]);
        ctx.fillStyle = 'rgb('+color.map((c,j)=>Math.round([38,52,56][j]*(1-v)+c*v)).join(',')+')';
        ctx.beginPath();ctx.arc(10+x*226,8+y*136,2,0,2*Math.PI);ctx.fill();
      });
    }""", dict(points=coords.tolist(), activity=activity, top=top, **{k: decision[k] for k in ('drop','action')}))


def annotate(pixels, box, detected, target, current, next_tier, decision, path):
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline='#2e7774', width=2)
    for x, y, tier, _ in detected:
        x, y = x0+x*(x1-x0), y0+y*(y1-y0)
        draw.ellipse((x-7,y-7,x+7,y+7), outline='white', width=2)
        draw.text((x+8,y-8), str(tier+1), fill='white', stroke_width=2, stroke_fill='#20302c')
    draw.line((target,y0,target,y1), fill='#2e7774', width=2)
    next_label = str(next_tier+1) if next_tier >= 0 else 'unknown'
    draw.rectangle((x0,y1+7,x1,y1+33), fill='#172e2d')
    draw.text((x0+8,y1+13), f'Drop {decision["drop"]}   Aim {decision["action"]+1}/15   Current {current+1}   Next {next_label}', fill='white')
    image.save(path, quality=82, optimize=True)


def run(args):
    graph = json.loads(args.graph.read_text())
    model, ckpt = restore(args.checkpoint, graph)
    if ckpt['kind'] != 'flywire' or not ckpt['updates']:
        raise ValueError('A trained FlyWire checkpoint is required')
    scorer = None
    if args.afterstate_checkpoint:
        from .afterstate_model import restore_scorer

        scorer = restore_scorer(args.afterstate_checkpoint,graph)
    rollout_models = None
    if args.rollout_mode or args.state_mode:
        from .afterstate_model import restore_scorer

        rollout_models = {k:restore_scorer(Path('artifacts/afterstate')/f'{k}.pt',graph)
                          for k in ('compact','flywire')}
    limit = args.wall_seconds if args.wall_seconds is not None else (None if args.until_game_over else 600)
    vision = Vision(args.cache,args.vision_workers,watermelon_inner=args.extended_vision)
    rng = np.random.default_rng(args.action_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    images = args.out / 'screens'
    images.mkdir(exist_ok=True)
    result = dict(url=URL, checkpoint=str(args.checkpoint), training_drops=ckpt['drops'],
                  updates=ckpt['updates'], seed=ckpt['seed'], action_seed=args.action_seed,
                  rules=ckpt.get('rules', 'original'), drop_cap=None if args.until_game_over else args.max_drops, time_cap_seconds=limit,
                  final_score=None, completed_drops=0, browser='headed Chromium', ad_filter=True, strategy=args.strategy, outcome='launch interrupted')
    if scorer:
        result.update(strategy=f'afterstate-{scorer.kind}',value_checkpoint=str(args.afterstate_checkpoint),
                      future_weight=args.future_weight,overflow_penalty=args.overflow_penalty,
                      activity_input='selected simulated afterstate' if scorer.kind=='flywire' else 'pre-drop diagnostic')
    if rollout_models:
        result.update(strategy=f'rollout-{args.rollout_mode}',rollout_mode=args.rollout_mode,
                      activity_input='pre-drop observation diagnostic',vision_workers=args.vision_workers)
    config = None
    if args.state_mode:
        config = json.loads(args.state_config.read_text())
        result.update(strategy=f'state-{args.state_mode}',state_config=str(args.state_config),
                      state_mode=args.state_mode,activity_input='pre-drop observation diagnostic')
    window = [] if args.state_mode in ('tracking','both') else None
    decisions = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='chromium', headless=False, args=['--disable-gpu'])
            page = browser.new_page(service_workers='block', viewport={'width':1280,'height':900})
            page.route('**/*', filter_ad_requests)
            started = time.monotonic()
            global RECORDER
            if args.record:
                RECORDER = Recorder(args.record, started)
            deadline = float("inf") if limit is None else started + limit
            page.goto(URL, wait_until='domcontentloaded', timeout=30000)
            board = page.locator('#game-container > canvas')
            bounds = board.bounding_box(timeout=12000)
            page.locator('#next-drop').wait_for(state='visible', timeout=12000)
            x, y, w, h = (bounds[k] for k in ('x','y','width','height'))
            box = tuple(round(v) for v in (x+.083*w, y+.164*h, x+.919*w, y+.973*h))
            x0,y0,x1,y1 = box
            pixels, hand, error = ready(page, vision, box, min(deadline, time.monotonic()+14),window)
            if error:
                page.screenshot(path=str(args.out/'paused.png'))
                result['outcome'] = error
                return
            background = pixels[y0:y1,x0:x1].copy()
            result['board_bounds'] = box
            extended = None
            if args.extended_vision:
                # Band background: the empty board, with spawn-band columns taken from three frames
                # whose held fruit sits elsewhere. Only the pointer moves; nothing is released.
                shots = []
                for sx in (x0+60, (x0+x1)/2, x1-60):
                    page.mouse.move(sx, y0-39)
                    page.wait_for_timeout(350)
                    image = screenshot(page)
                    image_hand, _ = hand_pixels(image, box)
                    found_hand = hand_position(vision.match(image_hand, range(5), (.64,.68,.72), (0,), .047), box)
                    if found_hand:
                        shots.append((found_hand[0], image))
                extended, uncovered = compose_background(pixels, shots, box)
                result.update(extended_vision=True, band_background_shots=len(shots), band_uncovered_columns=uncovered)
                if len(shots) < 2 or uncovered:
                    result['outcome'] = 'paused: spawn-band background incomplete'
                    page.screenshot(path=str(args.out/'paused.png'))
                    return
                # Re-read the board and held fruit together after the pointer moved.
                pixels = screenshot(page)
                hand_image, _ = hand_pixels(pixels, box)
                hand = vision.match(hand_image, range(5), (.64,.68,.72), (0,), .047)
                if not hand:
                    result['outcome'] = 'paused: held fruit not found after band background'
                    page.screenshot(path=str(args.out/'paused.png'))
                    return

            def detect_board(image, held, fast=False):
                if extended is None:
                    return vision.board(image[y0:y1,x0:x1], background, fast=fast)
                return board_extended(vision, image, box, extended, hand_position(held, box), fast=fast)

            while (args.until_game_over or len(decisions) < args.max_drops) and time.monotonic() < deadline:
                found, coverage = detect_board(pixels, hand, fast=True)
                if coverage < .86:
                    found, coverage = detect_board(pixels, hand)
                for _ in range(3):
                    if coverage >= .86 or time.monotonic() >= deadline:
                        break
                    page.wait_for_timeout(220)
                    pixels = settling_frame(page,window)
                    found, coverage = detect_board(pixels, hand)
                detected = [f for f in found if f[1] >= 0]
                band = [f for f in found if f[1] < 0]
                if coverage < .86:
                    result['outcome'] = f'paused on unreliable board detection ({coverage:.1%} explained foreground)'
                    page.screenshot(path=str(args.out/'paused.png'))
                    break
                nx = page.locator('#next-drop').bounding_box(timeout=1500)
                l,t = round(nx['x']-12),round(nx['y']-12)
                r,b = round(nx['x']+nx['width']+12),round(nx['y']+nx['height']+12)
                nxt = vision.match(pixels[t:b,l:r], range(5), (.48,.53,.58,.63,.68,.73), (0,), .05)
                next_tier = nxt[3] if nxt else -1
                current = hand[3]
                observation = observe([d[:3] for d in detected], current, next_tier)
                if args.state_mode:
                    from .state_search import decide as state_decide
                    from .state_estimation import fruits_from_obs

                    history = []
                    if window:
                        reference_time = window[-1][0]
                        for captured, prior in window[:-1]:
                            found, quality = vision.board(prior[y0:y1,x0:x1],background,fast=True)
                            if quality >= .86:
                                history.append((captured-reference_time,[r[:3] for r in found]))
                    history.append((0.,fruits_from_obs(observation)))
                    action, trace = state_decide(observation,history,model,rollout_models,args.state_mode,
                                                args.action_seed,len(decisions)+1,config,trace=True)
                elif rollout_models:
                    from .rollout_eval import rng_for
                    from .rollout_transfer import choose

                    drop = len(decisions)+1
                    action, trace = choose(observation,model,rollout_models,args.action_seed+drop,
                                           rng_for(args.action_seed,drop),args.rollout_mode)
                    _, diagnostic = model.decide(observation,None,'greedy')
                    trace.update(diagnostic)
                elif scorer:
                    action, trace = scorer.decide(observation,args.future_weight,args.overflow_penalty,trace=True)
                    if 'steps' not in trace:
                        _, diagnostic = model.decide(observation,rng,'greedy')
                        trace.update(diagnostic)
                else:
                    state = rng.bit_generator.state
                    action, trace = model.decide(observation, rng, args.strategy)
                    if args.contact_check:
                        from .planner_experiment import logp
                        from .robust_planner import choose as robust_choose
                        from .spawn_contact import gate, label_actions

                        labels = label_actions([(x*410, y*440, t) for x, y, t, _ in found], current, box)
                        blocked, allowed = gate(labels, action)
                        trace.update(legacy_action=action, blocked=blocked,
                                     contact_labels=[l['label'] for l in labels], legacy_contact=labels[action])
                        if blocked:
                            # Same random draws as the legacy call, shortlist restricted to non-fatal moves.
                            rng.bit_generator.state = state
                            action, _ = robust_choose(observation, logp(model, observation), rng, allowed=allowed)
                        trace['contact'] = labels[action]
                if extended is not None:
                    trace['band'] = [[round(x*410, 1), round(y*440, 1), int(t)] for x, y, t, _ in band]
                margin = vision.sprites[current].shape[1]*.34 + 3
                target = x0+margin+(x1-x0-2*margin)*AIMS[action]
                page.mouse.move(target,y0-39)
                page.wait_for_timeout(120)
                aimed = screenshot(page)
                error_px = None
                for attempt in range(4):
                    hand_image, left = hand_pixels(aimed, box)
                    check = vision.match(hand_image, [current], (.64,.68,.72), (0,), .047)
                    hand_x = left+check[1] if check else None
                    error_px = abs(hand_x-target) if check else None
                    if error_px is not None and error_px <= 13:
                        break
                    if page.locator('#restart-game').is_visible():
                        break
                    if attempt < 3:
                        page.mouse.move(target,y0-39)
                        page.wait_for_timeout(180)
                        aimed = screenshot(page)
                if page.locator('#restart-game').is_visible():
                    result['outcome'] = 'game over'
                    result['final_score'] = integer_text(page, '#final-score')
                    Image.fromarray(aimed).save(args.out/'end.png')
                    break
                if error_px is None or error_px > 13:
                    result['outcome'] = f'paused before clicking: aim verification failed ({error_px} px)'
                    page.screenshot(path=str(args.out/'paused.png'))
                    break
                number = len(decisions) + 1
                filename = f'screens/{number:03}.jpg'
                decision = dict(drop=number, action=action, target_x=target, aim_error_px=error_px,
                                coverage=coverage, observation=observation.tolist(), **trace,
                                score_before=integer_text(page, '#currentScore'), score=None,
                                image=filename, timestamp=time.monotonic()-started)
                activity_panel(page, graph, decision)
                captured = time.monotonic()
                aimed = screenshot(page)
                decision['hand_x'] = None if hand_x is None else float(hand_x)
                if RECORDER is not None:
                    RECORDER.drop = number
                    RECORDER.add('aim', (captured+time.monotonic())/2, aimed)
                if time.monotonic() >= deadline:
                    result['outcome'] = 'time cap reached before release'
                    Image.fromarray(aimed).save(args.out/'end.png')
                    break
                annotate(aimed, box, found, target, current, next_tier, decision, args.out/filename)
                if time.monotonic() >= deadline:
                    result['outcome'] = 'time cap reached before release'
                    (args.out/filename).rename(args.out/'time-cap.jpg')
                    break
                clicked = time.monotonic()
                page.mouse.click(target,y0-39)
                released = time.monotonic()
                decision['release_time'] = (clicked+released)/2-started
                decisions.append(decision)
                result['completed_drops'] = len(decisions)
                if RECORDER is None:
                    page.wait_for_timeout(750)
                else:
                    # Fill the same fixed 750 ms post-release wait with timestamped frames.
                    RECORDER.drop = number
                    cost = .05
                    while time.monotonic()-released+cost < .75:
                        captured = time.monotonic()
                        image = screenshot(page)
                        cost = time.monotonic()-captured
                        RECORDER.add('burst', captured+cost/2, image)
                    remaining = .75-(time.monotonic()-released)
                    if remaining > 0:
                        page.wait_for_timeout(remaining*1000)
                    decision['post_release_wait'] = time.monotonic()-released
                if window is not None:
                    window.clear()
                pixels, hand, error = ready(page, vision, box, min(deadline, time.monotonic()+18),window)
                decision['ready_time'] = time.monotonic()-started
                if args.until_game_over and error in ('board stability timed out', 'visible current fruit timed out'):
                    pixels, hand, error = recover_ready(page, vision, box, background, min(deadline,time.monotonic()+70),window,
                                                        None if extended is None else (lambda image, held: detect_board(image, held, True)))
                score = integer_text(page, '#final-score' if error == 'game over' else '#currentScore')
                decision['score'] = result['final_score'] = score
                if error:
                    if error != 'game over' and time.monotonic() >= deadline:
                        error = 'time cap reached'
                    result['outcome'] = error
                    page.screenshot(path=str(args.out/('end.png' if error in ('game over','time cap reached') else 'paused.png')))
                    break
            else:
                result['outcome'] = 'drop cap reached' if len(decisions) == args.max_drops else 'time cap reached'
                page.screenshot(path=str(args.out/'end.png'))
            result['seconds'] = time.monotonic()-started
            browser.close()
    except Exception as error:
        result['outcome'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if RECORDER is not None:
            RECORDER.close(dict(board_bounds=result.get('board_bounds'), outcome=result['outcome']))
        for name in ('end.png', 'paused.png', 'time-cap.jpg'):
            if (args.out/name).exists():
                result['final_capture'] = name
                break
        json_file(args.out/'manifest.json', dict(**result, drops=[
            {k:v for k,v in d.items() if k not in ('direct','steps','observation')} for d in decisions]))
        pack_script(args.out/'website.js', args.replay_key, dict(meta=result, decisions=decisions))
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, default=Path('artifacts/graph.json'))
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/flywire-11/final.pt'))
    parser.add_argument('--cache', type=Path, default=Path('data/site'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/demo'))
    parser.add_argument('--replay-key', default='website')
    parser.add_argument('--record', type=Path, help='save timestamped frames around each release')
    parser.add_argument('--extended-vision', action='store_true', help='detect the spawn band, upper corners and watermelons')
    parser.add_argument('--contact-check', action='store_true', help='block high-confidence spawn-contact game overs')
    parser.add_argument('--state-mode',choices=['current','tracking','physics','both'])
    parser.add_argument('--state-config',type=Path,default=Path('artifacts/state-estimation/frozen.json'))
    parser.add_argument('--rollout-mode',choices=['fixed','adaptive'])
    parser.add_argument('--vision-workers',type=int,default=8)
    parser.add_argument('--wall-seconds',type=int)
    parser.add_argument('--afterstate-checkpoint',type=Path)
    parser.add_argument('--future-weight',type=float,default=1.)
    parser.add_argument('--overflow-penalty',type=float,default=100.)
    parser.add_argument('--max-drops', type=int, choices=range(1,201), default=200)
    parser.add_argument('--until-game-over', action='store_true')
    parser.add_argument('--strategy', choices=['sample','planning','hybrid','lookahead','robust'], default='sample')
    parser.add_argument('--action-seed', type=int, default=32833)
    args = parser.parse_args()
    if args.contact_check and not args.extended_vision:
        parser.error('--contact-check requires --extended-vision')
    if args.contact_check and args.strategy != 'robust':
        parser.error('--contact-check is implemented for the robust planner')
    if args.vision_workers < 1:
        parser.error('--vision-workers must be positive')
    if sum(map(bool,(args.rollout_mode,args.afterstate_checkpoint,args.state_mode))) > 1:
        parser.error('Select one action scorer')
    if args.wall_seconds is not None and args.wall_seconds <= 0:
        parser.error('--wall-seconds must be positive')
    run(args)
