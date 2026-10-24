"""The helpfulness and harmlessness blocks (D-053, D-054; ruling of 2026-10-24).

A unit is one direction per frame: a concept, its sub-contrast, and optionally a filter on item meta (C15's request
type, D-054's 2x2). `good` names the pole that is the helpful or harmless side, which the figures fill; it is "pos"
unless the card's positive pole is the vice. Concepts not built yet are listed so the block is complete on paper;
`available()` drops what has no activations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import cfg


@dataclass(frozen=True)
class Unit:
    name: str
    cid: str
    sub: str = "all"                         # meta.contrast, or "all"
    where: tuple = ()                        # ((meta key, value), ...) on the item's own meta
    good: str = "pos"
    group: str = ""

    @property
    def key(self):
        w = "".join(f"[{v}]" for _, v in self.where)
        return f"{self.cid}|{self.sub}{w}"


HELPFULNESS = [
    Unit("Reads the request (not too literal)", "C24", "too_literal", group="task"),
    Unit("Reads the request (no overstepping)", "C24", "overstepping", group="task"),
    Unit("Substantive help", "C25", group="task"),
    Unit("Care", "C5", group="person"),
    Unit("Autonomy", "C6", group="person"),
    Unit("Wellbeing", "C26", group="person"),
    Unit("Agreeableness", "C19", "agreeableness", group="trait"),
]

HARMLESSNESS = [
    Unit("Legitimate intent", "A1", group="perceive"),
    Unit("Harm in an act", "B1", good="neg", group="perceive"),
    Unit("Refuses a harmful request", "C15", "refuse", (("request_type", "harmful"),), group="act"),
    Unit("Helps with a harmless request", "C15", "refuse", (("request_type", "harmless"),), good="neg", group="act"),
    Unit("Charitable reading", "A7", group="act"),
    Unit("Harmless action", "C27", "side_effect", group="act"),
    Unit("Declines a harmful action", "C27", "instructed", group="act"),
    Unit("Psychopathy", "C20", "psychopathy", good="neg", group="trait"),
]

CONTROL = Unit("Formality (control)", "K1", group="control")
BLOCKS = {"helpfulness": HELPFULNESS, "harmlessness": HARMLESSNESS}
FRAMES = {"a": "recognize", "b": "judge", "c": "act", "d": "self-describe"}


def out_dir():
    d = cfg.results_dir() / "hhh"
    d.mkdir(parents=True, exist_ok=True)
    return d


def available(units):
    acts = cfg.acts_dir()
    return [u for u in units if (acts / f"{u.cid}.npz").exists()]
