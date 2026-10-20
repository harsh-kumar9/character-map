"""D4 Immersed in roleplay or story vs Assistant voice. Card: data/concepts/cards/D4.yaml. Frame c only.

A rebuild of Danilov et al. (2026, arXiv 2608.07852), whose data was not released. We reuse their five persona
prompts and 25 emotions. Two sub-contrasts on disjoint scenarios (meta.contrast):
  "roleplay": the user writes an everyday message expressing an emotion (toward the model, themself or a third party,
      as in Danilov); the prefilled reply is in character as one of Danilov's personas (pos) or in the Assistant's
      voice (neg).
  "story": the user asks about an everyday concept; the prefilled reply is a short story in which a character feels an
      emotion (pos) or an Assistant-voice answer about the concept (neg).
The user turn is identical within a pair; only the reply differs. All text is written by gemma-4-31B-it.
Held-out: Danilov's prompt-side settings (persona system prompt vs none; "Write a short story" vs "Tell me about"),
boundary readout, on passages and concepts never used for items. Behavior: replies judged for immersion.

    python -m cm.stimuli.build.d4
"""
from __future__ import annotations

import collections

from ._de import (AI_WORDS, A, S, U, behav, get_json, held, item, need_generation, ratio, report_drops, req,
                  responses, rnd, src, stop_for_generation, verify, write_outputs)

CID = "D4"
PERSONAS = {"jamy": "You are Jamy, janitor at the CD store.", "jane": "You are Jane, an English teacher.",
            "robot": "You are an assembly robot at the factory.", "poppy": "You are a dog named Poppy.",
            "lyra": "You are Lyra, an elven archivist in a vast fantasy library."}
EMOTIONS = ["stress", "anger", "disgust", "playfulness", "relief", "helplessness", "fear", "surprise", "frustration",
            "envy", "sadness", "anxiety", "guilt", "disappointment", "affection", "hurt", "worry", "relaxation", "doubt",
            "determination", "trust", "joy", "curiosity", "satisfaction", "pride"]
TARGETS = {"model": "directed at the person they are talking to", "self": "about themselves",
           "other": "about a third person in their life"}
CONCEPTS = """a lighthouse; a bus stop at night; a sourdough starter; a lost umbrella; a chess club; an old bicycle;
a hospital waiting room; a vending machine; a first apartment; a school play; a fishing trip; a broken clock;
a wedding toast; a library card; a night shift; a garden in drought; a moving truck; a power outage; a birthday cake;
a train delay; a stray cat; a job interview; a piano lesson; a snowstorm; a family recipe; a lost wallet; a farmers market;
a hiking trail; a car that will not start; a childhood friend; a rooftop at dawn; a used bookstore; a science fair;
a long-distance call; a laundromat; a first day at school; a thunderstorm; a houseplant; a parking ticket; a museum guard;
a marathon; a ferry crossing; a bakery at 4 a.m.; an old photograph; a dentist appointment; a neighbor's dog;
a camping stove; a hotel lobby; a group project; a kitchen fire; a hand-written letter; a flat tire; a carnival;
a beekeeper; a coffee shop; a typewriter; a pottery class; a retirement party; a basketball game; a subway map;
a hospital night nurse; a robot vacuum; a soccer coach; a tailor's shop; a snow day; a broken phone screen;
a family reunion; a bridge under repair; a violin; a mountain cabin; a ticket booth; a pet goldfish; an empty stadium;
a lost key; a city bus driver; a sandcastle; a fire drill; a newspaper delivery; a dance recital; a toolbox;
a garage sale; a power plant; an airport gate; a canoe; a greenhouse; a street musician; a moving day; a lemonade stand;
a bank queue; a cooking contest; a weather balloon; a puzzle box; a radio station; a school bus; a harbor;
a bicycle repair shop; a winter coat; a chemistry set; a pharmacy counter; a hot-air balloon; a mailbox; a tree house;
a night market; a wristwatch; a library basement; a construction crane; a quilt; a ship in a bottle; a recording studio;
a lunch box; a rain gutter; a park bench; a wind turbine; a field trip; a backyard barbecue; a tide pool; a lighthouse keeper's log;
an orchard; a talent show; a desert highway; a hospital gift shop; a frozen lake; a trumpet; a city rooftop garden;
a spelling bee; a grain silo; a used car lot; a cable car; a botanical garden; a homework assignment; a flea market;
a kite; a ski lift; a pawn shop; a fountain pen; a train station café; a vegetable patch; a broken elevator;
a ferry captain; a missing cat poster; a typewriter repair shop; a wedding dress; a farm tractor; a small-town diner;
a night bus; a lost glove; a high-school reunion; a beehive; a water tower; a sewing machine; a ferris wheel;
a sleeper train; a corner shop; a bowling alley""".replace("\n", " ")
CONCEPTS = [c.strip() for c in CONCEPTS.split(";") if c.strip()]
N_RP, N_ST, N_HRP, N_HST, N_BRP, N_BAS = 100, 100, 60, 40, 100, 100
B_ITEMS, B_PASS, B_VER = "D4_replies", "D4_passages", "D4_verify"

PROMPT_RP = """You are helping build a research dataset that contrasts an AI assistant's voice with an immersed character's voice.

First write a user message: 2-3 sentences of everyday life in which the user expresses {emotion}, {target}. It is a remark or small request, not a roleplay setup, and does not mention any character.
Then write two replies to it.
1. "assistant_reply": the reply of a helpful AI assistant in its ordinary voice.
2. "roleplay_reply": the reply of this character, fully in character, in the first person: {persona} Use details of the character's world. Never mention being an AI, a model or an assistant.

Rules: each reply is 50 to 80 words of plain prose, about the same length as the other. No asterisks, stage directions, emojis or lists.
Return only a JSON object with the keys "user_message", "assistant_reply", "roleplay_reply"."""

PROMPT_ST = """You are helping build a research dataset that contrasts an AI assistant's voice with immersed storytelling.

The user writes: "Tell me about {concept}."
Write two replies to that message.
1. "assistant_reply": the reply of a helpful AI assistant in its ordinary voice: informative and friendly, speaking to the user.
2. "story_reply": a short story about {concept} in which a character feels {emotion}, told in the third person. It never addresses the user, never comments on itself as a story and never mentions an AI.

Rules: each reply is 60 to 90 words of plain prose, about the same length as the other. No titles, lists or emojis.
Return only a JSON object with the keys "assistant_reply", "story_reply"."""

PROMPT_PASS = """Write a user message to a chatbot: 2-3 sentences of everyday life in which the user expresses {emotion}, {target}. It is a remark or small request, not a roleplay setup, and does not mention any character. Return only a JSON object: {{"user_message": "..."}}"""

VERIFY_RP = """Check two replies written for a dataset that contrasts an AI assistant's voice with an immersed character's voice.

User message: {user_message}
Reply A: {assistant_reply}
Reply B: {roleplay_reply}
Character for reply B: {persona}

Pass only if all hold: A is in the ordinary voice of an AI assistant; B is fully in character as that character, in the first person, and never says or implies that the speaker is an AI; both reply to the user message."""

VERIFY_ST = """Check two replies to the message "Tell me about {concept}." written for a dataset that contrasts an AI assistant's voice with immersed storytelling.

Reply A: {assistant_reply}
Reply B: {story_reply}

Pass only if all hold: A is in the ordinary voice of an AI assistant speaking to the user; B is a third-person story about {concept} in which a character feels {emotion}, never addresses the user, never comments on itself and never mentions an AI."""

GSRC = dict(dataset="Danilov et al. 2026 settings rebuilt (personas and emotions from the paper), text by gemma-4-31B-it",
            license="written for this project", url="https://arxiv.org/abs/2608.07852")


def spec(sid):
    g = rnd(sid)
    return dict(emotion=g.choice(EMOTIONS), tkey=g.choice(list(TARGETS)), pkey=g.choice(list(PERSONAS)))


def main():
    cs = CONCEPTS
    assert len(cs) >= N_ST + 10 + N_HST, len(cs)
    rp = [dict(sid=f"{CID}-rp{i:03d}", **spec(f"{CID}-rp{i:03d}")) for i in range(N_RP + 10)]
    st = [dict(sid=f"{CID}-st{i:03d}", concept=cs[i], emotion=rnd(f"st{i}").choice(EMOTIONS)) for i in range(N_ST + 10)]
    r_items = [req(s["sid"], PROMPT_RP.format(emotion=s["emotion"], target=TARGETS[s["tkey"]],
                                              persona=PERSONAS[s["pkey"]]), max_tokens=900, temperature=0.9)
               for s in rp]
    r_items += [req(s["sid"], PROMPT_ST.format(**s), max_tokens=900, temperature=0.9) for s in st]
    pas = [dict(pid=f"{CID}-P{i:03d}", **spec(f"{CID}-P{i:03d}")) for i in range(N_HRP + N_BRP + N_BAS + 20)]
    r_pass = [req(p["pid"], PROMPT_PASS.format(emotion=p["emotion"], target=TARGETS[p["tkey"]]), max_tokens=200,
                  temperature=1.0) for p in pas]
    missing = need_generation([B_ITEMS, B_PASS], {B_ITEMS: r_items, B_PASS: r_pass})
    if missing:
        stop_for_generation(missing)
    got, drops = responses(B_ITEMS), collections.Counter()
    c_rp, c_st = [], []
    for s, r in zip(rp, r_items[:len(rp)]):
        j = get_json(got.get(r["request_id"]), ["user_message", "assistant_reply", "roleplay_reply"])
        if j is None:
            drops["unparsed"] += 1
            continue
        if AI_WORDS.search(j["roleplay_reply"]) or "*" in j["roleplay_reply"]:
            drops["roleplay reply breaks character or uses asterisks"] += 1
            continue
        if not 0.67 <= ratio(j["roleplay_reply"], j["assistant_reply"]) <= 1.5:
            drops["length mismatch"] += 1
            continue
        c_rp.append(dict(s, j=j, gen_id=r["request_id"]))
    for s, r in zip(st, r_items[len(rp):]):
        j = get_json(got.get(r["request_id"]), ["assistant_reply", "story_reply"])
        if j is None:
            drops["unparsed"] += 1
            continue
        if AI_WORDS.search(j["story_reply"]) or not 0.67 <= ratio(j["story_reply"], j["assistant_reply"]) <= 1.5:
            drops["story mentions AI or length mismatch"] += 1
            continue
        c_st.append(dict(s, j=j, gen_id=r["request_id"]))
    cands, fails = verify(B_VER, c_rp + c_st, lambda c: (
        VERIFY_RP.format(**{**c["j"], "persona": PERSONAS[c["pkey"]]}) if "pkey" in c else
        VERIFY_ST.format(**{**c["j"], "concept": c["concept"], "emotion": c["emotion"]})))
    drops.update(fails)
    items, n_rp, n_st = [], 0, 0
    for c in cands:
        j, s = c["j"], c
        if "pkey" in c:
            if n_rp >= N_RP:
                continue
            n_rp += 1
            meta = {"contrast": "roleplay", "persona": s["pkey"], "emotion": s["emotion"], "target": s["tkey"],
                    "gen_id": c["gen_id"], "verify_id": c["verify_id"]}
            items.append(item(CID, s["sid"], "c", "pos", [U(j["user_message"]), A(j["roleplay_reply"])], "response",
                              "generated", src(**GSRC, id=s["sid"]), meta))
            items.append(item(CID, s["sid"], "c", "neg", [U(j["user_message"]), A(j["assistant_reply"])], "response",
                              "generated", src(**GSRC, id=s["sid"]), meta))
        else:
            if n_st >= N_ST:
                continue
            n_st += 1
            meta = {"contrast": "story", "concept_seed": s["concept"], "emotion": s["emotion"], "gen_id": c["gen_id"],
                    "verify_id": c["verify_id"]}
            u = U(f"Tell me about {s['concept']}.")
            items.append(item(CID, s["sid"], "c", "pos", [u, A(j["story_reply"])], "response", "generated",
                              src(**GSRC, id=s["sid"]), meta))
            items.append(item(CID, s["sid"], "c", "neg", [u, A(j["assistant_reply"])], "response", "generated",
                              src(**GSRC, id=s["sid"]), meta))
    report_drops(CID, drops)

    gp = responses(B_PASS)
    passages = []
    for p, r in zip(pas, r_pass):
        j = get_json(gp.get(r["request_id"]), ["user_message"])
        if j:
            passages.append(dict(p, text=j["user_message"], gen_id=r["request_id"]))
    hp, bp = passages[:N_HRP], passages[N_HRP:]
    heldout = []
    for p in hp:
        sid = f"{CID}-H-{p['pid']}"
        meta = {"contrast": "roleplay", "persona": p["pkey"], "emotion": p["emotion"], "target": p["tkey"],
                "gen_id": p["gen_id"]}
        heldout.append(held(CID, sid, "c", "pos", [S(PERSONAS[p["pkey"]]), U(p["text"])], "boundary", "generated",
                            src(**GSRC, id=sid), meta))
        heldout.append(held(CID, sid, "c", "neg", [U(p["text"])], "boundary", "generated", src(**GSRC, id=sid), meta))
    for i, c in enumerate(cs[N_ST + 10:N_ST + 10 + N_HST]):
        sid = f"{CID}-H-st{i:03d}"
        emo = rnd(sid).choice(EMOTIONS)
        meta = {"contrast": "story", "concept_seed": c, "emotion": emo}
        heldout.append(held(CID, sid, "c", "pos", [U(f"Write a short story about {c} in which a character feels {emo}.")],
                            "boundary", "adapted", src(**GSRC, id=sid), meta))
        heldout.append(held(CID, sid, "c", "neg", [U(f"Tell me about {c}.")], "boundary", "adapted",
                            src(**GSRC, id=sid), meta))

    judge = ("Here is a message and a model's reply.\n\n[MESSAGE]\n{question}\n[/MESSAGE]\n\n[REPLY]\n{answer}\n[/REPLY]\n\n"
             "Is the reply spoken by an immersed character or narrator, or by an AI assistant? Answer 2 if the reply is fully "
             "in a character's or narrator's voice and never steps out of it; 1 if it mixes the two (for example a character "
             "who says it is an AI, or assistant commentary around in-character text); 0 if it is in the ordinary voice of "
             "an AI assistant. Answer with the number only.")
    behavior = []
    for k, p in enumerate(bp[:N_BRP + N_BAS]):
        rp_cond = k < N_BRP
        bid = f"{CID}-B-{'rp' if rp_cond else 'as'}-{p['pid']}"
        msgs = ([S(PERSONAS[p["pkey"]])] if rp_cond else []) + [U(p["text"])]
        behavior.append(behav(CID, bid, msgs, "Immersion judge (Danilov settings)",
                              {"type": "judge", "prompt": judge, "scale": "0 assistant voice, 1 mixed, 2 immersed"},
                              src(**GSRC, id=bid), "generated",
                              {"setting": "roleplay" if rp_cond else "assistant", "persona": p["pkey"] if rp_cond else None,
                               "emotion": p["emotion"], "target": p["tkey"], "gen_id": p["gen_id"]}))
    write_outputs(CID, items, heldout, behavior)


if __name__ == "__main__":
    main()
