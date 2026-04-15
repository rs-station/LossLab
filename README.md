# LossLab: Modular Coordinate Refinement Library

LossLab is a modular library supporting coordinate refinement against experimental data: cryo-EM maps, crystallographic structure factors, and beyond.

LossLab is based on pytorch.


## Structure and Vision

LossLab implements two primary abstractions:

1. **Losses.** These are likelihood functions that compute the probability of some structure given a set of experimental data: `p(x|D)`. A common interface to these losses is enforced by an abstract base class, `BaseLoss`.

2. The **Refinement Engine**, a gradient decent manager and logger. Many of the outputs of refinement are common to all refinement strategies: structures as a function of iteration, compute metrics, etc. The `RefinementEngine` class implements these common features and provides a foundation which specific refinement implementations can extend.


## Out of scope

LossLab does not generate or sample structures/coordinates. LossLab simply provides a likelihood (and, via torch, liklihood gradients) and a generic system for tracking progress as one seeks to optimize that likelihood.

LossLab assumes your are working with a discrete list of cartesian coordinates that represent atomic positions. Models that use densities, continous distributions, _etc._ are out of scope.
