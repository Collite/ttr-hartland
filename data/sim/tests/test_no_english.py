"""The NH-D8 guard itself — later stages run every rendered label, note, letter and sheet through `assert_czech`."""

import unicodedata

import pytest

from hartland_sim.checks.czech import LONG_TEXT, assert_czech, assert_czech_all, english_words, has_diacritics

CZECH = [
    # a complaint letter — carries the words Czech shares with English: a, i, to, do, on, no, by
    "Dobrý den, píšu vám kvůli poslední dodávce na naši prodejnu v Kolíně. Ze tří kartonů Harcafé Gold byly "
    "dva poškozené a jeden chyběl úplně. Řidič to i tak potvrdil a odjel do Prahy. No, on to stejně podepsal. "
    "Prosím o dobropis k faktuře FV-2025-004812, jinak bychom museli objednávky omezit, aby se to neopakovalo. "
    "Děkuji, Jana Nováková",
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
    "poskozene a jeden chybel uplne. Ridic to potvrdil a odjel do Prahy. Prosim o dobropis k fakture "
    "FV-2025-004812, jinak budeme objednavky omezovat."
)

# Loanwords Czech writes hyphenated (review-188 R1).
HYPHENATED_LOANWORDS = [
    "Roll-up Harcafé Gold u pokladny",
    "Předali jsme know-how",
    "Check-in v hotelu",
    "Tiskárna all-in-one",
]

# Invented names only — the repo is public.
STORE_LETTER = "Reklamace z provozovny The Corner Bistro: dodávka nedorazila."

ENGLISH_LABELS = [
    "Order confirmed",
    "Customer name",
    "Total amount due",
    "Delivery address",
    "Invoice number",
    "Payment terms",
    "Unit price",
    "Discount applied",
    "Product description",
    "Shipping method",
    "Warehouse location",
    "Account manager",
    "Credit limit exceeded",
    "Return reason",
    "Complaint status",
]

CZECH_LABELS = [
    "Objednávka potvrzena",
    "Jméno zákazníka",
    "Celková částka k úhradě",
    "Dodací adresa",
    "Číslo faktury",
    "Platební podmínky",
    "Jednotková cena",
    "Uplatněná sleva",
    "Popis výrobku",
    "Způsob dopravy",
    "Umístění skladu",
    "Obchodní zástupce",
    "Překročený úvěrový limit",
    "Důvod vrácení",
    "Stav reklamace",
]


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


@pytest.mark.parametrize("text", HYPHENATED_LOANWORDS)
def test_a_hyphenated_loanword_is_one_token_and_never_a_stop_word(text: str) -> None:
    assert english_words(text) == []
    assert_czech(text)


def test_allow_exempts_the_names_a_renderer_filled() -> None:
    with pytest.raises(AssertionError, match=r"English words \['the'\]"):
        assert_czech(STORE_LETTER)
    assert_czech(STORE_LETTER, allow=["The Corner Bistro"])
    assert english_words(STORE_LETTER, allow=["The Corner Bistro"]) == []


def test_allow_exempts_names_not_the_sentence_around_them() -> None:
    with pytest.raises(AssertionError, match="English words"):
        assert_czech("The delivery is late.", allow=["Corner"])


def test_all_capitals_english_fails() -> None:
    with pytest.raises(AssertionError, match=r"English words \[.*'the'.*\]"):
        assert_czech("PLEASE NOTE: THE DELIVERY IS DELAYED")


@pytest.mark.parametrize("text", ["IT oddělení", "IT ODDĚLENÍ"])
def test_a_listed_acronym_is_not_a_word(text: str) -> None:
    assert english_words(text) == []
    assert_czech(text)


def test_a_capitalised_word_is_still_a_word() -> None:
    assert english_words("It works") == ["it"]


def test_a_column_of_short_english_labels_fails_as_a_whole() -> None:
    for label in ENGLISH_LABELS:
        assert_czech(label)  # each one alone is too short to tell — the limit the docstring states
    assert len(" ".join(ENGLISH_LABELS)) >= LONG_TEXT
    with pytest.raises(AssertionError, match=r"labels: 15 texts, \d+ characters without a Czech diacritic"):
        assert_czech_all(ENGLISH_LABELS, what="labels")


def test_a_column_of_czech_labels_passes() -> None:
    assert_czech_all(CZECH_LABELS, what="labels")


def test_assert_czech_all_names_the_failing_text() -> None:
    with pytest.raises(AssertionError, match=r"labels\[1\]: English words \['the'\]"):
        assert_czech_all(["Jméno zákazníka", "The name"], what="labels")


def test_nfd_text_fails_as_not_nfc() -> None:
    nfd = unicodedata.normalize("NFD", CZECH[0])
    with pytest.raises(AssertionError, match="letter: not NFC"):
        assert_czech(nfd, what="letter")
    assert_czech(unicodedata.normalize("NFC", nfd))
