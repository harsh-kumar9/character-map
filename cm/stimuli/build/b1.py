"""B1 care/harm: Moral Stories pairs by foundation, MFV held-out (see _foundations.py).

    python -m cm.stimuli.build.b1
"""
from ._foundations import build

def main():
    build("care")


if __name__ == "__main__":
    main()
