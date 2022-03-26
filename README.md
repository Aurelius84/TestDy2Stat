# TestDy2Stat
More Unittests for Deep Learning Framework in Dynamic2Static Feature.

## What's this?

## TensorFlow
`@tf.function` is an important feature for TF 2.x and it support transform users' imperative model codes into Static Graph for offline predicting.

## PyTorch
Torch introduces `@jit.trace` and `@jit.script` for users, which exports models as `jit::ScriptModule` to be easily loaded by libtorch.

## Goals
This repository aims to deeply explore the mechanism of them by writing many unittests to see how they implement this detaily step by step.
