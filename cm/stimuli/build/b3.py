"""B3 loyalty/betrayal: Moral Stories pairs by foundation, MFV held-out (see _foundations.py).

    python -m cm.stimuli.build.b3
"""
from ._foundations import build


def main():
    build("loyalty")


if __name__ == "__main__":
    main()
