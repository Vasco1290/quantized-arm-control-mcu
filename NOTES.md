# Kote — design notes

One short paragraph per decision: what we chose and why. This becomes the README and the answers to interview questions.

---

## Physics you need (only this much)

**Weight is a pull.** Gravity pulls every object down with a force of mass × 9.81, measured in newtons (N). Roughly: 1 kg pulls with about 10 N.

**Torque is turning force.** A joint does not care about a straight pull, only about how hard something tries to *rotate* it: torque = pull × sideways distance from the joint, in N·m. A door is easy to push at the handle and nearly impossible next to the hinge. Example: a 1 kg bag held 0.5 m out from the shoulder gives 10 × 0.5 = 5 N·m; pull it in to 0.25 m and the torque halves to 2.5 N·m. With the arm hanging straight down, every part is directly below the shoulder, so the sideways distance and the gravity torque are both zero. Straight out forward, the distance and the torque are at their largest.

**Motors make torque.** Our motor command (ctrl) goes from −1 to +1 and the gear multiplies it: torque = ctrl × gear. With gear 30, ctrl 1 gives 30 N·m.

**sin in the arm formula.** Tilt a 1 m stick from hanging straight down: sin(angle) is how far forward its tip reaches (sin 0° = 0, sin 90° = 1). The hand formula x = L1·sin(q1) + L2·sin(q1+q2) reads: how far forward the elbow is, plus how much further forward the hand is beyond the elbow. The forearm uses q1+q2 because the elbow angle is measured from the upper arm, not from straight down.

---

## Warm-up and how we judge controllers

**Warm-up on Reacher first.** Before building our own arm, we ran Gymnasium's Reacher-v5 (a flat 2-joint arm, no gravity) to learn the reset/step loop and get baselines: random −41.83 ± 4.47, do-nothing −11.86 ± 4.80 (10 seeds).

**Reward alone can be gamed.** On Reacher, doing nothing scored *better* than random, even though neither reaches the target, because Reacher's reward punishes motor effort and random wastes effort. So a good reward does not mean a good controller. Every controller in this project is judged on, in this order: (1) success rate, (2) final hand–target distance, (3) reward, kept only as a secondary number.

**Success = within 3 cm at the end, fixed in advance.** An episode counts as a success if the hand ends within 3 cm of the target's centre after 1 second (the target ball itself has a 2.5 cm radius, so this means "touching it"). We use the *final* distance, not the best moment, because a controller that passes through the target and drifts away has not reached it. The threshold was picked before any learned controller existed and is never changed (`SUCCESS_DIST` in `sim/exo_arm_env.py`), so it cannot be tuned to flatter a result.

**Same 100 seeded episodes for everyone.** Every controller plays seeds 0..99. The seed fixes the start pose and the target, so differences between rows come from the controller, not luck, and anyone can re-run the test and get the same numbers.

---

## The arm model (`sim/exo_arm.xml`)

**Why our own arm instead of Reacher.** Reacher has no gravity, so holding still is free. A real arm, and an exoskeleton helping one, spends most of its effort fighting gravity. Our arm moves in the vertical plane with gravity on, so the controller has to hold the arm up as well as move it.

**Two joints in the vertical plane.** Shoulder and elbow only, both hinges, moving forward/back and up/down. Angles are measured from hanging straight down, and positive means swinging forward. Shoulder range −30..150°, elbow 0..145°.

**Elbow never bends backwards (0..145°).** A human elbow cannot hyper-extend. It also makes the maths unique: a 2-joint arm can usually reach a point in two mirror-image ways (elbow bent one way or the other), and allowing only one bending direction leaves exactly one answer for inverse kinematics.

**Human-sized links.** Upper arm 0.33 m / 2.0 kg, forearm + hand 0.26 m / 1.5 kg. Each is modelled as a uniform capsule. This is a simplification, because real limbs are heavier near the body.

**Exoskeleton mass rides on the arm, except the shoulder motor.** In real exoskeletons the shoulder motor bolts to a frame on the wearer's back, so the arm never lifts it. The elbow motor has to sit at the elbow, so the upper arm carries it. We modelled it that way. Adding the exo parts raised the shoulder's gravity torque (arm straight out) from 10.0 to 12.5 N·m, and the 0.35 kg elbow motor alone causes about 1.1 N·m of that (0.35 × 9.81 × 0.33 m), because it sits far from the shoulder. This is why exo designers try to keep motors close to the body.

**Motor strength: gear 30 (shoulder), 15 (elbow).** Holding the arm straight out needs ctrl 0.42 at the shoulder (12.52 ÷ 30) and 0.15 at the elbow (2.18 ÷ 15). That leaves room to move the arm, not just hold it.

**Physics settings.** Time step 0.002 s with the `implicitfast` integrator (stable with joint damping). Small joint damping (0.5) and armature (0.01) stand in for friction and motor inertia. Collisions are off: the arm cannot hit the body. This is a known limitation, acceptable because targets are kept in front of and around the wearer.

---

## The environment (`sim/exo_arm_env.py`)

**Same shape as Reacher.** The env follows Gymnasium's `reset`/`step` template and gives the same 10 observation numbers in the same order as Reacher, so Reacher-style code and habits carry over.

**50 decisions per second, 1-second episodes.** The physics runs in 0.002 s ticks. The controller decides every 10 ticks (0.02 s) and its command is held in between. 50 Hz is a normal rate for arm control, and it leaves time for a real chip on WiFi to answer (a 2 ms budget would be impossible). Before fixing 1 s, a rough teacher was checked to reach 100/100 targets within it (worst miss 2 mm), so the time limit is fair.

**Start: arm resting.** Each episode starts with the arm hanging down (shoulder −5..+5°, elbow 0..10°), like a wearer at rest.

**Targets made from joint angles.** Instead of picking a random point in space (which might be unreachable), we pick random joint angles and use the arm formula to find where the hand would be. The ball goes there, so every target is reachable. The angle ranges (shoulder −20..140°, elbow 10..135°) stay a little inside the real limits, so the arm never has to push against a hard stop to reach a target.

**Observation: angles as cos/sin.** An angle has a jump: 179° and −179° point almost the same way but are far apart as numbers. cos and sin have no jump, and neural networks learn much more easily from smooth inputs. The other numbers are target position, joint speeds (so the controller can brake), and hand-minus-target (which way to go). Positions are measured from the shoulder, which keeps the numbers small.

**Reward = −distance − 0.1 × effort.** Effort is each motor command squared, added together. The effort weight is kept small (0.1) because this arm has to spend effort just to stay up: a large effort penalty would reward letting the arm hang, which is the same gaming problem we saw on Reacher. Reward stays a secondary metric anyway.

**The teacher may peek, the student may not.** Controllers are functions `(obs, env) -> action`. The teacher (PD + gravity compensation) is allowed to read the true simulator state, including gravity torques. The student network later sees only the 10 observation numbers.

---

## The teacher (`sim/teacher.py`)

**Why a classical teacher, not reinforcement learning.** A hand-built controller that we fully understand gives a known-good "ceiling" in one evening, with no training. The student network then only has to *copy* it (behaviour cloning), which is much faster and more reliable than learning by trial and error. Any drop in quality after that is caused by the network, quantization or hardware, not by a weak teacher.

**Three parts, every 0.02 s.** (1) Inverse kinematics turns the target position into the two joint angles that put the hand there. (2) PD pulls each joint toward its goal angle. (3) Gravity compensation adds exactly the torque gravity is taking away. The sum, divided by the gear, is the motor command.

**Inverse kinematics in two steps.** Elbow first: the upper arm, the forearm and the line from shoulder to target form a triangle, and the law of cosines gives the elbow angle from the three side lengths (far target = straighter elbow). Then the shoulder: point the upper arm at the target, then lean back by the extra angle the bent elbow adds. Only the elbow-forward answer is used, because our elbow cannot bend the other way. Checked on 20,000 random poses: going angles → position → angles returns the same angles (error ~1e-10°).

**Gravity compensation = making the arm weightless.** MuJoCo can tell us, at any pose, exactly how much torque gravity is putting on each joint (`qfrc_bias`, which also includes small effects from the arm's own speed). Adding that much torque cancels gravity, so the arm would float still wherever it is. The PD part then only has to move a "weightless" arm. This is the part the teacher is allowed to "peek" at: the student never sees `qfrc_bias` and has to learn the effect from the 10 observation numbers.

**PD = a spring plus a brake.** P (proportional): push = KP × (goal angle − current angle). It is like a spring: far from the goal it pushes hard, close to it gently. D (derivative): push = −KD × joint speed. It is like a shock absorber: it resists motion, so the arm slows down as it arrives instead of overshooting and wobbling.

**Gains: KP = 60 / 20, KD = 8 / 2 (shoulder / elbow), from a scan.** We tried the spring at 0.5×, 1×, 1.5×, 2.5× and the brake at 0.5×, 1×, 2× (100 seeds each). The chosen setting reached 100% with an average miss of 0.03 cm, arriving in ~0.37 s, with the motor at its limit on 8% of steps. A stiffer spring also worked (still 100% at 2.5×) but maxed out the motor more often (20%). A **brake twice as strong failed badly** (0–55% success). The reason: the command is held for 0.02 s between updates, and a very strong brake pushes so hard during that hold that it reverses the motion; the next update brakes the other way, and the arm chatters. Our chosen brake is only about 2× away from that edge. Extra delay (WiFi, a slow chip) makes the hold effectively longer, so we expect delay to hurt in the same way, which the delay sweep will test.

**Commands are clipped to −1..+1 inside the teacher.** The env would clip them anyway, but the student learns from the teacher's outputs, so those outputs must be the commands that were really applied, not impossible ones like 2.3.

**Inverse kinematics needs an angle wrap.** An early version once returned −230° for the shoulder. That is the same direction as +130° taken the long way round, so the controller drove the shoulder into its −30° stop. The fix: always wrap angles into −180..180°.

---

## The student (`sim/collect.py`, `sim/train_student.py`, `sim/student.py`)

**Behaviour cloning, not reinforcement learning.** The student learns by copying: we record what the teacher does, then train a network to give the same answer for the same observation. It is ordinary supervised learning (like learning to label pictures from examples), so it trains in under a minute, gives the same result every run, and cannot discover odd reward-gaming tricks the way trial-and-error learning can.

**Practice targets never overlap test targets.** The teacher is recorded on seeds 1000..2999 (2,000 episodes, 100,000 observation→action pairs). Testing uses seeds 0..99. So the student's score is always on targets it has never seen.

**Wobble while recording (noise injection).** While recording, the arm is driven with the teacher's command plus a small random error (spread 0.1 on the −1..+1 scale), but we save the teacher's *clean* command. This pushes the arm slightly off course, so the recording also contains the teacher *correcting* mistakes. Without it, the student only ever sees perfect trajectories; the moment it makes a tiny error of its own, it is somewhere it has never seen and its errors pile up. Measured: a student trained on clean data had a *lower* training error but controlled the arm worse (average miss 0.27 cm, worst 1.73 cm) than the wobble-trained one (0.11 cm, worst 0.60 cm). Lesson: copying error on recorded data is not the same as control quality; only the closed-loop test counts.

**Network: 10 → 64 → 64 → 2, ReLU.** Two hidden layers of 64 is plenty for a smooth 10-input function (it learns inverse kinematics plus gravity compensation implicitly) and has only 4,994 numbers (weights + biases), so INT8 it is about 5 KB, which fits easily on both the Pi and the ESP32. ReLU (keep positives, zero out negatives) is the cheapest activation to run and to quantize: no exponentials, just a comparison. The output layer has no activation; the command is clipped to −1..+1 afterwards, like the teacher's.

**Input scaling.** Joint speeds can be around 10 while positions are around 0.5. Before the network, each input is shifted by its average and divided by its typical spread (both measured on the training data and saved with the weights), so all ten inputs sit around −1..+1. Networks learn much faster when inputs are on similar scales, and it will make quantization easier because one number range fits all inputs.

**Training settings.** Loss = mean squared difference from the teacher's command. Adam optimiser, learning rate 1e-3 slowly lowered to 0 (cosine schedule), batches of 256, 60 passes over the data, fixed random seed. The last 10% of episodes are held back for validation: validation error (0.00025) stayed close to training error (0.00022), so the network learned the pattern rather than memorising examples.

**PyTorch trains, NumPy runs.** The trained weights are saved as plain NumPy arrays (`models/student_float.npz`, committed so results are reproducible without retraining). `sim/student.py` runs the network with three hand-written multiply-and-add steps. This exact code is what gets quantized to INT8 and copied to the Pi, so the float and INT8 rows differ only by quantization. Checked: NumPy and PyTorch outputs match to within 7e-7.

---

## Results so far (exo arm, 100 seeds)

| Controller | Success | Final distance | Reward |
|---|---|---|---|
| random | 0% | 54.58 ± 29.40 cm | −31.85 ± 12.94 |
| do-nothing | 0% | 65.76 ± 28.70 cm | −32.83 ± 14.35 |
| PD teacher | 100% | 0.03 ± 0.05 cm | −6.87 ± 4.08 |
| float student | 100% | 0.11 ± 0.09 cm | −6.88 ± 4.08 |

Random and do-nothing never reach a target: the floor. The teacher reaches all of them: the ceiling. The float student matches the teacher on success, misses by ~1 mm on average (worst 0.6 cm), so copying the teacher into a network costs almost nothing. It leaves a 2.4 cm safety margin under the 3 cm line for quantization and delay to eat into. Every later row (INT8, on-chip) is measured against this one. Unlike Reacher, doing nothing does not win on reward here, because the effort penalty is small.
