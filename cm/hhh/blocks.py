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
    short: str = ""                          # figure label, naming the good side

    @property
    def label(self):
        return self.short or self.name

    @property
    def key(self):
        w = "".join(f"[{v}]" for _, v in self.where)
        return f"{self.cid}|{self.sub}{w}"


HELPFULNESS = [
    Unit("Reads the request (not too literal)", "C24", "too_literal", group="task", short="Not too literal"),
    Unit("Reads the request (no overstepping)", "C24", "overstepping", group="task", short="No overstepping"),
    Unit("Asks when a request is ambiguous", "C28", "under_asks", group="task", short="Asks if ambiguous"),
    Unit("Answers when a request is clear", "C28", "over_asks", group="task", short="Answers if clear"),
    Unit("Substantive help", "C25", group="task", short="Substantive"),
    Unit("Care", "C5", group="person", short="Care"),
    Unit("Autonomy", "C6", group="person", short="Autonomy"),
    Unit("Wellbeing", "C26", group="person", short="Wellbeing"),
    Unit("Agreeableness", "C19", "agreeableness", group="trait", short="Agreeable"),
]

HARMLESSNESS = [
    Unit("Legitimate intent", "A1", group="perceive", short="Legit intent"),
    Unit("Harm in an act", "B1", good="neg", group="perceive", short="Act harmless"),
    Unit("Refuses a harmful request", "C15", "refuse", (("request_type", "harmful"),), group="act", short="Refuses harmful"),
    Unit("Helps with a harmless request", "C15", "refuse", (("request_type", "harmless"),), good="neg", group="act", short="Helps harmless"),
    Unit("Charitable reading", "A7", group="act", short="Charitable"),
    Unit("Harmless action", "C27", "side_effect", group="act", short="Harmless action"),
    Unit("Declines a harmful action", "C27", "instructed", group="act", short="Declines action"),
    Unit("Psychopathy", "C20", "psychopathy", good="neg", group="trait", short="Low psychopathy"),
]

# read only, for the cross-block figures (D-055): the honesty block as the honesty pages draw it (D-049)
HONESTY = [
    Unit("Self-knowledge", "A9", group="epistemic"),
    Unit("Accuracy", "C2", group="epistemic"),
    Unit("Sincerity", "C1", "lying", group="character"),
    Unit("Forthrightness", "C3", group="character"),
    Unit("Non-sycophancy", "C8", "sycophantic_agreement", group="character"),
    Unit("Honesty-humility", "C19", "honesty_humility", group="trait"),
]

CONTROL = Unit("Formality (control)", "K1", group="control")
BLOCKS = {"helpfulness": HELPFULNESS, "harmlessness": HARMLESSNESS, "honesty": HONESTY}
FRAMES = {"a": "recognize", "b": "judge", "c": "act", "d": "self-describe"}


def out_dir():
    d = cfg.results_dir() / "hhh"
    d.mkdir(parents=True, exist_ok=True)
    return d


def available(units):
    acts = cfg.acts_dir()
    return [u for u in units if (acts / f"{u.cid}.npz").exists()]
