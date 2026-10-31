"""The NH-D8 guard itself — later stages run every rendered label, note, letter and sheet through `assert_czech`."""

import pytest

from hartland_sim.checks.czech import LONG_TEXT, assert_czech, english_words, has_diacritics

CZECH = [
    # a complaint letter — carries the words Czech shares with English: a, i, to, do, on, no, by
    "Dobrý den, píšu vám kvůli poslední dodávce na naši prodejnu v Kolíně. Ze tří kartonů Harcafé Gold byly dva "
    "poškozené a jeden chyběl úplně. Řidič to i tak potvrdil a odjel do Prahy. No, on to stejně podepsal. Prosím o "
    "dobropis k faktuře FV-2025-004812, jinak bychom museli objednávky omezit, aby se to neopakovalo. Děkuji, Jana "
    "Nováková",
    # a promotion sheet — ten, my, and the acronym IT
    "Promoce PR-2025-11-GOLD-TV: Harcafé Gold 200 g se slevou 15 % ve všech řetězcích od 10. listopadu do "
    "7. prosince. Podpora v televizi, letáky i vystavení na čele regálu. Ten, kdo splní cíl, dostane bonus; my "
    "zajistíme materiály. IT oddělení připraví kódy do pokladen.",
    # a visit note — short; most (a bridge)
    "Regál doplněn. Kavalír má akci na zimní kávu, most přes Labe je zavřený, dodávky jezdí objížďkou.",
]

ENGLISH = [
    "Dear customer, thank you for your order. The delivery will arrive on Monday.",
    "We are writing to inform you that the shipment of instant coffee scheduled for this week has been delayed "
    "because of a vehicle breakdown near the distribution centre. Our dispatch team will contact you as soon as a "
    "new delivery window is confirmed. We apologise for any inconvenience.",
    "Děkujeme za objednávku, please confirm the delivery date.",
]

CZECH_WITHOUT_ACCENTS = (
    "Dobry den, pisu vam kvuli posledni dodavce na nasi prodejnu v Koline. Ze tri kartonu Harcafe Gold byly dva "
    "poskozene a jeden chybel uplne. Ridic to potvrdil a odjel do Prahy. Prosim o dobropis k fakture FV-2025-004812, "
    "jinak budeme objednavky omezovat."
)


@pytest.mark.parametrize("text", CZECH)
def test_czech_passes(text: str) -> None:
    assert_czech(text)


@pytest.mark.parametrize("text", ENGLISH)
def test_english_fails(text: str) -> None:
    with pytest.raises(AssertionError, match="English words"):
        assert_czech(text)


def test_the_message_names_the_words_and_the_text() -> None:
    with pytest.raises(AssertionError, match=r"letter 7: English words \['please', 'the'\] in 'Děkujeme za"):
        assert_czech(ENGLISH[2], what="letter 7")
    assert english_words(ENGLISH[2]) == ["please", "the"]


def test_long_text_without_diacritics_fails() -> None:
    assert len(CZECH_WITHOUT_ACCENTS) >= LONG_TEXT
    assert english_words(CZECH_WITHOUT_ACCENTS) == []
    with pytest.raises(AssertionError, match="without a Czech diacritic"):
        assert_czech(CZECH_WITHOUT_ACCENTS)


def test_short_text_without_diacritics_passes() -> None:
    assert not has_diacritics("Harsoup Klasik 60 g")
    assert_czech("Harsoup Klasik 60 g")


def test_slovak_passes() -> None:
    assert_czech("Dobrý deň, posielam reklamáciu k faktúre, tovar prišiel poškodený a chcem ho vrátiť.")


def test_acronyms_are_not_words_but_capitalised_words_are() -> None:
    assert english_words("IT oddělení") == []
    assert english_words("It works") == ["it"]
