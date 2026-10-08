"""Heuristic code-switching CANDIDATE detection.

This is deliberately conservative and is NOT a language identifier. It flags tokens that
belong to small hand-written lists of common English and Swahili words, after removing
words that collide with frequent Dholuo words (e.g. Dholuo "an" = I, "in" = you,
"ok" = not, "to" = and/but, "bed" = to be/stay, "ber" = good).

Flags are candidates only. The audit notebook writes them to
``codeswitch_candidates.csv`` with an empty ``verified`` column; a Dholuo speaker must
confirm them before Experiment E reports anything as "code-switched". Results computed
from unverified flags must be labelled as such.
"""
from __future__ import annotations

ENGLISH_COMMON = set("""
the and for are but not you all any can had her was one our out day get has him his how
man new now old see two way who boy did its let put say she too use that with have this
will your from they know want been good much some time very when come here just like long
make many more only over such take than them well were what school church hospital doctor
phone money bank market teacher student office police government county president
computer internet radio television mobile business company water food health family
because about after again also back before could even first give into most other people
should still their there these thing think those through under where which while would
year years week month today tomorrow yesterday please thank thanks sorry okay yes
""".split())

SWAHILI_COMMON = set("""
sana lakini kwa kwamba hapana ndiyo asante karibu habari leo kesho jana saa shule
kanisa hospitali daktari mwalimu serikali pesa soko simu basi tafadhali pole sawa
kweli wewe mimi yeye sisi nyinyi wao hii hiyo ile hapa pale kila bado tena pia ndio
""".split())

# Frequent Dholuo words that collide with entries above (or with short English words).
DHOLUO_COLLISIONS = set("""
an in to ok no en ne ni ka ma mar gi ga be bed ber man wan kod gin mos gik ten dok nam
kit yot pod tim ting kata koso mano ere ang' ok kae kanyo bende
""".split())

ENGLISH_LEX = ENGLISH_COMMON - DHOLUO_COLLISIONS
SWAHILI_LEX = SWAHILI_COMMON - DHOLUO_COLLISIONS


def tag_tokens(tokens: list[str]) -> list[str]:
    """Return one of {'en','sw','luo?'} per token ('luo?' = not flagged)."""
    out = []
    for t in tokens:
        t = t.lower()
        if t in ENGLISH_LEX:
            out.append("en")
        elif t in SWAHILI_LEX:
            out.append("sw")
        else:
            out.append("luo?")
    return out


def codeswitch_candidates(tokens: list[str]) -> list[str]:
    return [t for t, tag in zip(tokens, tag_tokens(tokens)) if tag != "luo?"]
