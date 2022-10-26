# Playing 6 nimmt! with deep reinforcement learning

[6 nimmt!](https://en.wikipedia.org/wiki/6_nimmt!) is a card game for 2 to 10 
players designed in 1994. It is a fun game that involves risk management in face of 
uncertainty, and a great bit of chance.

This repository contains the code and the trained artificial neural networks (ANN)
for this game (for 2/4/5 players). These trained ANNs are very competitive players,
definitely better than I am :/

There are two scripts if you are willing to play against them (I recommend 
using real-world cards for visual representation). The first is `human_against_machines.py`
in which a human is invited to challenge `n` number of ANN foes. The other,
`machine_against_humans.py` can be used to challenge your friends.

To set up the environment, you can install the dependencies from the project's 
root directory with

```bash
$ python install .
```

and then you can run any of these two scripts (see `--help` option for details).

I'm also leaving a quick presentation (see [25-10-2022 - Team meeting - take6.pptx](25-10-2022%20-%20Team%20meeting%20-%20take6.pptx))
I made to Amadeus' research team about some design decisions.
