"""B2 fairness/cheating: Moral Stories pairs by foundation, MFV held-out (see _foundations.py).

    python -m cm.stimuli.build.b2
"""
from ._foundations import build


def main():
    build("fairness")


if __name__ == "__main__":
    main()
