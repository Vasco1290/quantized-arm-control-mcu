# Running the controller on the Pi Zero W

The laptop simulates the arm; the Pi runs the network. Everything below was tested with a
stand-in "Pi" on the laptop (`pi/pi_server.py` on 127.0.0.1), so on the day only the
address changes.

## 1. Things you need

- USB microSD card reader (the laptop's SD slot doesn't fit the adapter)
- A micro-USB cable that carries **data**, not just power (for power only, any cable works once WiFi is set up)
- The Pi Zero W only has **2.4 GHz WiFi**: the laptop and the Pi must be on a 2.4 GHz network (or the same router)

## 2. Flash the card (on the laptop)

Raspberry Pi Imager → **Raspberry Pi OS Lite (32-bit)** → choose the card → edit settings:

- hostname `kote-pi`
- username and password (your choice)
- WiFi: network name, password, country **IN**
- Services: **enable SSH** (password login)

Put the card in the Pi, power it, wait ~2 minutes for the first boot.

## 3. Check you can reach it

```bash
ping kote-pi.local
ssh <user>@kote-pi.local
```

If `kote-pi.local` doesn't resolve, find the Pi's IP address in the router's device list and use that instead everywhere below.

## 4. Install NumPy on the Pi (on the Pi)

```bash
sudo apt update
sudo apt install -y python3-numpy
mkdir -p ~/kote/models
```

Use `apt`, not `pip`: the Pi Zero's processor (ARMv6) has no ready-made pip packages, and building NumPy on it takes hours.

## 5. Copy the files (on the laptop, from the repo folder)

```bash
scp pi/pi_server.py pi/protocol.py sim/int8_net.py sim/student.py <user>@kote-pi.local:~/kote/
scp models/student_int8.npz models/student_delay1.npz models/student_delay2.npz models/student_delay3.npz models/student_delay4.npz models/student_delay5.npz <user>@kote-pi.local:~/kote/models/
```

## 6. Start the server (on the Pi)

```bash
cd ~/kote && python3 pi_server.py
```

It prints the controllers it loaded and `listening on UDP port 5005`. Leave it running.

## 7. Run the tests (on the laptop)

Each run takes about 2 minutes (100 episodes of 1 s in real time) and adds a row to `results/hw/summary.csv`.

```bash
# 1. Correctness + raw timing: must match the laptop INT8 row (100%, 0.27 cm). Read the round-trip times.
.venv/Scripts/python.exe sim/hw_loop.py --host kote-pi.local --controller int8 --mode lockstep

# 2. Realistic loop, plain INT8 student, answers used as they arrive
.venv/Scripts/python.exe sim/hw_loop.py --host kote-pi.local --controller int8 --mode asap

# 3. Fixed-delay buffer + matching delay-trained student. Pick k so that k x 20 ms is above
#    the 99% round-trip time from run 1 (e.g. 99% = 14 ms -> k = 1; 99% = 33 ms -> k = 2).
.venv/Scripts/python.exe sim/hw_loop.py --host kote-pi.local --controller delay1 --mode buffer --k 1

# 4. Extra lost messages on top of the real network (repeat run 2 and 3 with these)
.venv/Scripts/python.exe sim/hw_loop.py --host kote-pi.local --controller int8 --mode asap --loss 0.1
.venv/Scripts/python.exe sim/hw_loop.py --host kote-pi.local --controller delay1 --mode buffer --k 1 --loss 0.3
```

## If something doesn't work

- **"no answer from the chip"**: is `pi_server.py` running on the Pi? Same WiFi network? Try the IP address instead of `kote-pi.local`.
- **Windows firewall**: replies to messages the laptop sent are normally allowed. If not, allow Python for private networks when Windows asks.
- **Timing looks bad**: keep the Pi close to the router, and close other heavy programs on the laptop (the summary prints "laptop overruns" if the laptop itself fell behind).
