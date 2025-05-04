# Connect-Zero: Reinforcement Learning from Scratch

![alt text](https://c-f-h.github.io/post/connect-zero/screenshot.png#center)

This repository implements Reinforcement Learning (RL) techniques from
scratch for the game of Connect 4.
It serves as the companion repo to the [blog post series about
Connect-Zero](https://c-f-h.github.io/post/connect-zero/).
Currently it implements:

- basic REINFORCE
- REINFORCE with baseline
- A2C (Actor-Critic with TD(1) value bootstrapping)

It also contains some utility scripts for having models play single
games or tournaments against each other, perform pretraining,
evaluate the performance on tactical puzzles, and export models to
ONNX format.

The scripts require ``torch``, ``matplotlib``, ``numpy``, and ``click``.

To run the examples, navigate to the ``train/`` subdirectory and
execute e.g.

    $ python example3-rwb.py

The ``webapp/`` subdirectory contains a JavaScript applet for
interactively playing against an ONNX exported model.
