# AI for radar: background for tasks 6–9

This page is shared background for four tasks, and a list of further directions. It is not a
task itself. Read it, pick **one** of [the four tasks](#pick-one-task) or propose one of
[the directions](#directions), and read [the intern README](README.md) for setup, house rules
and how to submit.

**AI, machine learning and deep learning are not currently implemented in `radar-forge`.** There
is no learned model, no training code and no PyTorch import anywhere in `src/`. You will build
what you need in your own script, on top of what the library already produces. As always, leave
`src/` alone. If you think the library needs to change, say why in your report. That is a
finding.

## What "AI enhancement" means here

Each task puts one learned or statistical block next to the classical block it would replace,
and asks one question: **does it do better, on a problem we control, at equal cost in false
alarms?** Papers that claim a gain often compare against a weak or mistuned baseline, at a
different false-alarm rate, or on the data the model was trained on. Your job is to avoid all
three.

Some of these methods are not deep learning at all. Extreme value theory and conformal
prediction are statistics, and they are here because they answer questions a learned radar
block raises: how do you set a threshold for a false-alarm rate you can barely measure, and how
do you know a model's uncertainty is honest?

## What the library does not have

Know this before you choose. Each item is something you may have to build in your own script.

- **No clutter.** The background is thermal noise only. Any clutter is clutter you add yourself,
  with parameters justified from a cited source on real radar data.
- **One target per scenario, one receive channel.** Two targets, or crossing tracks, are yours to
  synthesise. `radar_forge.core.propagation.line_of_sight_paths` takes arrays of ranges, velocities
  and RCS, so one call can place several targets in one IQ cube.
- **One tracker.** A UKF with constant-velocity or constant-acceleration motion
  (`radar_forge.core.tracking.motion.CartesianMotion`), a chi-square gate, and global or greedy
  nearest-neighbour assignment. There is no IMM, no PDA or JPDA, no MHT and no smoother.
- **No resource manager.** Nothing schedules beams, revisits or dwells.

## Rules for every task and direction

The [track-before-detect background](track-before-detect.md#definitions) defines SNR, false-track
probability and detection. Use those definitions, or say where and why you depart from them.

- **Predict first.** Write down one falsifiable hypothesis with a number in it, and what result
  would refute it, before you run anything.
- **Same false-alarm rate.** Compare every detector at the same Pfa or false-track probability,
  set on held-out noise-only (or clutter-only) runs. A detector that wins at a higher false-alarm
  rate has not won.
- **A good baseline.** Tune the classical method with the same care, and on the same data, as
  yours.
- **Separate seeds.** Keep training, threshold-setting and test data apart: random seeds, and
  windows of the trajectory.
- **Shift.** Test under at least one change from the training conditions: a different SNR,
  clutter parameter, part of the flight or manoeuvre. Report how fast any advantage disappears.
- **Cost.** Report training time, and operations and memory per frame at inference, against the
  baseline.
- **Small models.** Anything learned should train in minutes on a laptop CPU. A large model that
  wins is a weaker result than a small one that wins.

PyTorch is in the `ml` extra. `uv sync` removes every extra you do not name, so name them all:

```bash
uv sync --extra dev --extra viz --extra ml
```

## Pick one task

| # | Task | The question | Time | PyTorch |
| :--- | :--- | :--- | :--- | :--- |
| 6 | [Can an unrolled solver beat a taper?](unrolled-sparse-rd.md) | Does a learned, unrolled sparse solver separate two close targets better than a Taylor taper, at the same false-peak rate? | about 2–3 days | yes |
| 7 | [Does the tracker know its own uncertainty?](conformal-tracking.md) | Is the UKF's covariance honest, and when it is not, can a conformal gate or interval restore the coverage it promises? | about 2 days | stretch only |
| 8 | [Thresholds from the tail](evt-thresholds.md) | How few noise or clutter samples does an extreme-value fit need to set a reliable threshold at Pfa = 1e-6? | about 2 days | stretch only |
| 9 | [Learned data association](learned-association.md) | With crossing targets and false alarms, does a small graph network beat global nearest neighbour at the same false-track rate? | about 3 days | yes |

## Directions

The table below lists further directions, each with a classical baseline and a question. You
may propose one instead of tasks 6–9. **Agree it with us before you start.** Then follow the
deliverables of [task 5](world-models.md#deliverables): a short report, one script, an AI-use
note and Bugs and suggestions.

The "Build first" column is what the library lacks for that direction. Count it in your time.

| Technique | What it is | Baseline | Question | Build first |
| :--- | :--- | :--- | :--- | :--- |
| Differentiable simulation ("inverse rendering") | Fit scene parameters by gradient descent through a differentiable copy of the forward model | FFT peak with interpolation; the Cramér-Rao lower bound (CRLB) | Fit range, velocity and RCS through a PyTorch copy of the FMCW IQ model. Does the fit reach the CRLB, and where does it stick in local minima? | The PyTorch IQ model, checked against `fmcw_deramp_baseband` |
| Normalizing flows | An invertible network with an exact likelihood | CA-CFAR and OS-CFAR; the likelihood-ratio test that knows the true clutter law (a ceiling, not a rival) | Is a flow-learned clutter density, used as a likelihood-ratio detector, better than CFAR at equal Pfa? Does it keep its Pfa when the clutter level changes, as CFAR does? | Clutter; see [task 8](evt-thresholds.md) |
| Uncertainty quantification | Separating aleatoric from epistemic uncertainty, and checking calibration | NEES and NIS of the library's UKF | Does a deep ensemble of a learned motion model flag a flight regime it was not trained on, before its errors grow? | The learned motion model |
| Neural SDE | A learned drift and diffusion in continuous time | `CartesianMotion` (constant velocity, constant acceleration) | The ADS-B fixes are 2–8 s apart. Does a neural SDE handle the irregular spacing better than a constant-velocity model with a time-scaled process noise, with an honest covariance? | Overlaps [task 5, direction A](world-models.md#the-questions) |
| Mixture of experts | A gating network that weights specialist models | An IMM filter: Markov switching between motion models | Does a learned gate switch between motion models faster or more accurately than IMM's Markov chain? | An IMM, about one day |
| Transformer | Attention over a sequence | Constant-velocity Kalman prediction | What does attention over 30 s of track history buy over a constant-velocity filter, at each prediction horizon? The data is one aircraft: say what that limits. | — |
| Foundation model | A large pretrained model used without training, or fine-tuned | Constant-velocity and constant-acceleration Kalman prediction | Does a pretrained time-series model forecast the ADS-B track better than a Kalman filter, per operation? | **Ask us first.** Pretrained weights come from outside, and their use needs our agreement |
| Physics-informed network | A network trained with a physics residual in its loss | A Rauch–Tung–Striebel (RTS) smoother | The RTS smoother is optimal when its motion model is right. Does a kinematics-constrained network reconstruct the ADS-B path better when the model is wrong? | An RTS smoother |
| Reinforcement learning | A policy learned from reward by trial and error | Fixed revisit rate; covariance-triggered revisit (Van Keuk and Blackman) | Can a learned policy hold track with less energy on target than covariance-triggered revisits? | A revisit environment |
| Cognitive radar | A perception–action loop: the radar adapts what it transmits to what it believes (Haykin) | Always one waveform | Choose the waveform each frame by expected information gain from the tracker's posterior, with no learning. How close does it get to a learned policy? | Overlaps [task 5, direction D](world-models.md#the-questions) |

Diffusion models and self-supervised pretraining are not on the list, for now. A diffusion
model trained on our one aircraft's track has nothing to teach the tracker that the same data
cannot teach directly. And self-supervised pretraining pays off when labels are scarce, which
they are not in a simulation that labels every frame.

## Sources

The rule is the one in the [track-before-detect background](track-before-detect.md#sources):
cite a source for every architecture, loss, clutter model, threshold method and metric you use,
in your report and in the `References` section of the docstring that implements it, and read
what you cite. Take particular care with this topic. AI tools describe these methods fluently,
overstate what they guarantee, and invent citations. 
