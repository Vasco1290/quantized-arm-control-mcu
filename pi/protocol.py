"""Message formats between the laptop and the chip. Shared by both sides; copy to the Pi.

Sent over UDP (no waiting for lost messages to be re-sent: a late motor command is useless,
so it is better to lose it). Every message is a few fixed-size binary numbers, little-endian.

  laptop -> chip   RESET : type, episode number, which controller to use
  chip -> laptop   ACK   : type, episode number                      (reset received)
  laptop -> chip   OBS   : type, episode number, step, 10 observation numbers
  chip -> laptop   CMD   : type, episode number, step, 2 motor commands, compute time (microseconds)

The episode number lets each side ignore stray messages left over from a previous episode.
The step number lets the laptop match each answer to the observation it answers.
"""
import struct

PORT = 5005
OBS, RESET = 0, 1  # message types (CMD reuses 0, ACK reuses 1: direction tells them apart)

# Controller codes: 0 = plain INT8 student, k = delay-trained student built for k steps.
CONTROLLERS = {"int8": 0, "delay1": 1, "delay2": 2, "delay3": 3, "delay4": 4, "delay5": 5}

_OBS = struct.Struct("<BII10f")    # B = 1-byte int, I = 4-byte unsigned int, f = 4-byte float
_CMD = struct.Struct("<BII2fI")
_RESET = struct.Struct("<BIB")
_ACK = struct.Struct("<BI")


def pack_obs(episode, step, obs):
    return _OBS.pack(OBS, episode, step, *[float(v) for v in obs])


def pack_cmd(episode, step, command, compute_us):
    return _CMD.pack(OBS, episode, step, float(command[0]), float(command[1]), compute_us)


def pack_reset(episode, controller_code):
    return _RESET.pack(RESET, episode, controller_code)


def pack_ack(episode):
    return _ACK.pack(RESET, episode)


def unpack_to_chip(data):
    """Chip side: returns ("obs", episode, step, obs) or ("reset", episode, code), or None."""
    if len(data) == _OBS.size and data[0] == OBS:
        _, ep, step, *obs = _OBS.unpack(data)
        return ("obs", ep, step, obs)
    if len(data) == _RESET.size and data[0] == RESET:
        _, ep, code = _RESET.unpack(data)
        return ("reset", ep, code)
    return None


def unpack_to_laptop(data):
    """Laptop side: returns ("cmd", episode, step, (c1, c2), compute_us) or ("ack", episode), or None."""
    if len(data) == _CMD.size and data[0] == OBS:
        _, ep, step, c1, c2, us = _CMD.unpack(data)
        return ("cmd", ep, step, (c1, c2), us)
    if len(data) == _ACK.size and data[0] == RESET:
        _, ep = _ACK.unpack(data)
        return ("ack", ep)
    return None
