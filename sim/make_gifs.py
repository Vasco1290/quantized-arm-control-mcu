"""Make the README GIFs from the side camera.

  media/hero.gif         INT8 student reaching a few targets, real time
  media/delay_compare.gif  60 ms delay: plain student (left) vs delay-trained student (right),
                         same targets, half speed so the wobble is easy to see
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from delay_student import DelayStudent
from evaluate import run_episode
from exo_arm_env import ExoArmEnv, SUCCESS_DIST
from int8_net import int8_policy
from student import student_policy

MEDIA = Path(__file__).resolve().parent.parent / "media"
CROP = (60, 20, 540, 400)   # part of the 640x480 render that contains the person and the full reach
HOLD_FRAMES = 12            # frames to pause on the result at the end of each reach

HERO_SEEDS = [5, 2, 18, 3, 11]
COMPARE_SEEDS = [5, 3, 18]
COMPARE_DELAY = 3           # steps of 20 ms = 60 ms


def font(size):
    for name in ("seguisb.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def record(policy, seed, delay_steps=0):
    """Play one episode and return (frames, final distance)."""
    env = ExoArmEnv(render_mode="rgb_array")
    frames = []
    dist, _ = run_episode(env, policy, seed, delay_steps=delay_steps,
                          on_frame=lambda: frames.append(Image.fromarray(env.render()).crop(CROP)))
    env.close()
    return frames, dist


def label(img, lines, xy=(14, 10), size=18):
    """Write text with a dark shadow so it reads on the light and dark parts of the scene."""
    d = ImageDraw.Draw(img)
    f = font(size)
    x, y = xy
    for text in lines:
        d.text((x + 1, y + 1), text, font=f, fill=(0, 0, 0))
        d.text((x, y), text, font=f, fill=(255, 255, 255))
        y += size + 6
    return img


def result_text(dist):
    return f"reached ({100 * dist:.1f} cm)" if dist < SUCCESS_DIST else f"missed by {100 * dist:.1f} cm"


def save_gif(frames, path, frame_ms):
    # A GIF can hold only 256 colours. One shared colour table for all frames keeps the file
    # small and stops colours flickering; it is built from frames spread over the whole GIF,
    # so small things like the red target keep their colour.
    picks = frames[:: max(1, len(frames) // 8)]
    w, h = frames[0].size
    sheet = Image.new("RGB", (w, h * (len(picks) + 1)))
    for i, f in enumerate(picks):
        sheet.paste(f, (0, i * h))
    # The target ball is tiny, so on its own its red would be merged into the orange motor
    # rings. Add a strip of its shades (sampled from a render) so the red survives.
    for j, shade in enumerate([(243, 54, 63), (200, 40, 52), (255, 95, 100), (160, 35, 45)]):
        sheet.paste(shade, (j * w // 4, len(picks) * h, (j + 1) * w // 4, (len(picks) + 1) * h))
    palette_src = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    small = [f.quantize(palette=palette_src, dither=Image.Dither.NONE) for f in frames]
    small[0].save(path, save_all=True, append_images=small[1:], duration=frame_ms, loop=0, optimize=True)
    print(f"saved {path}  ({len(frames)} frames, {path.stat().st_size / 1e6:.1f} MB)")


def hero():
    out = []
    for seed in HERO_SEEDS:
        frames, dist = record(int8_policy, seed)
        frames = frames[::2]  # 25 frames per second of sim time, shown at real speed
        for f in frames:
            out.append(label(f.copy(), ["INT8 network (5.4 KB)"]))
        out += [label(frames[-1].copy(), ["INT8 network (5.4 KB)", result_text(dist)])] * HOLD_FRAMES
    save_gif(out, MEDIA / "hero.gif", frame_ms=40)


def compare():
    out = []
    for seed in COMPARE_SEEDS:
        left, d_left = record(student_policy, seed, COMPARE_DELAY)
        right, d_right = record(DelayStudent(COMPARE_DELAY), seed, COMPARE_DELAY)
        head_l = ["plain student", "60 ms delay"]
        head_r = ["delay-trained student", "60 ms delay"]
        w, h = left[0].size

        def pair(a, b, extra_l=(), extra_r=()):
            img = Image.new("RGB", (2 * w + 6, h), (20, 20, 20))
            img.paste(label(a.copy(), head_l + list(extra_l)), (0, 0))
            img.paste(label(b.copy(), head_r + list(extra_r)), (w + 6, 0))
            label(img, ["half speed"], xy=(2 * w + 6 - 110, h - 30), size=15)
            return img

        for a, b in zip(left, right):  # every step, shown at 40 ms each = half speed
            out.append(pair(a, b))
        out += [pair(left[-1], right[-1], [result_text(d_left)], [result_text(d_right)])] * (2 * HOLD_FRAMES)
    save_gif(out, MEDIA / "delay_compare.gif", frame_ms=40)


if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    compare()
    hero()
