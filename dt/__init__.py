"""Decision Transformer for CartPole (CS 446)."""

# CartPole-v1's step limit; also the scale applied to returns-to-go.
MAX_EP_LEN = 500
RTG_SCALE = 500.0
# Observation dims kept when velocities are hidden: [x, theta].
POSITION_DIMS = [0, 2]
