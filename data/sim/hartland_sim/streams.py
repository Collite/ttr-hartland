"""The committed catalogue of named random streams (contracts §2).

One name per kind of draw; `rng()` refuses a name not listed here, and a test asserts every `rng(` call site in the
package names one by its literal. Contracts §2 lists the first twenty-one; `skills` (the actors' hidden skills, S1.1
T4) and `promo` (prolong's promotions, S5.3 T4) are review-196 ⚑C's, `org` (the people's names and dates, S1.1 T4)
is S1.1's. A stage that needs a new kind of draw adds its name here — adding one never shifts another's draws.
"""

from __future__ import annotations

STREAMS: frozenset[str] = frozenset(
    {
        "demand",  # latent demand per customer × line × day (S1.2)
        "basket",  # order triggers and basket line sets (S1.2)
        "acquire",  # potential → active; the opening customers' `since` (S1.1, S1.2)
        "churn",  # active → lost (S1.2)
        "visit",  # visit adherence, times, outcomes (G2 S2.1)
        "note",  # the visit notes' templates and slots (G2 S2.1)
        "observe",  # the visit observations (G2 S2.1)
        "sighting",  # competitor sightings (G2 S2.1)
        "prompt_outcome",  # a rep prompt's outcome (G2 S2.2)
        "pos",  # PoS placements, give-aways, the asset pool (S1.1, G2 S2.3)
        "pricing_agree",  # agreed discounts and individual prices (S1.3)
        "payment",  # how and when a customer pays (G4 S4.1)
        "return",  # returns (G4 S4.1)
        "complaint",  # complaints (G4 S4.1, DOC.2)
        "supply",  # reorders, receipts, supplier reliability (S1.1, G3 S3.1)
        "route",  # the route plans and travel times (S1.1, G3 S3.2)
        "event",  # delivery events (G3 S3.3)
        "shop",  # the own shops' receipts (G4 S4.2)
        "weather",  # the daily weather series (S1.2)
        "letter",  # the letters' variants and injected errors (DOC.2)
        "scan",  # the scanned letters (DOC.2)
        "skills",  # the reps', managers' and KAMs' hidden skills (S1.1)
        "promo",  # promotions generated past the committed calendar (S5.3)
        "org",  # the people: names, hiring dates (S1.1)
    }
)
