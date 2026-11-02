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

## Baselines on the exo arm (100 seeds)

| Controller | Success | Final distance | Reward |
|---|---|---|---|
| random | 0% | 54.6 ± 29.4 cm | −31.85 ± 12.94 |
| do-nothing | 0% | 65.8 ± 28.7 cm | −32.83 ± 14.35 |

Neither reaches a single target: this is the floor of the results table. Unlike Reacher, doing nothing does not win on reward here, because the effort penalty is small.

---

## Things learned while testing (for the teacher)

**Inverse kinematics needs an angle wrap.** A rough IK once returned −230° for the shoulder. That is the same direction as +130° taken the long way round, so the controller drove the shoulder into its −30° stop. The fix is to always wrap angles into −180..180°.

**Stiffer is not better at 50 Hz (preliminary).** In a rough test, a gentle PD controller reached 100/100 targets, but stiffer settings reached only about 33–44%. The likely reason is that the command is only updated every 0.02 s, so a stiff controller overshoots between updates. If true, extra delay (WiFi, a slow chip) should make this worse, which is exactly what the delay sweep will measure.
