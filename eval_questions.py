"""Evaluation question set for the 14 CFR (FAA regulations) RAG chatbot.

30 questions that probe the corpus from many angles so we can see how the system
holds up beyond the 5 originally-prepared questions:

  - answerable, single-section (one CFR Part/§)
  - answerable, cross-section / comparison (needs 2+ sections)
  - ambiguous phrasing (exercises query rewrite)
  - out-of-corpus but aviation (e.g. drones — should say "not covered")
  - off-topic noise (should return the no-info response)

Each item:
  q              the question
  category       grouping label for the report
  answerable     True if the answer is in the corpus (expect a grounded answer);
                 False if it should return no-info / "not covered"
  expect_sources index source filenames whose content should ground a good answer
                 (empty for out-of-corpus questions)

Keep this list stable so before/after comparisons stay fair.
"""

P61 = "CFR-2025-title14-vol2-part61.pdf"
P67 = "CFR-2025-title14-vol2-part67.pdf"
P71 = "CFR-2025-title14-vol2-part71.pdf"
P73 = "CFR-2025-title14-vol2-part73.pdf"
P91 = "CFR-2025-title14-vol2-part91.pdf"

QUESTIONS = [
    # ── Part 61 — pilot certification ────────────────────────────
    {"q": "What aeronautical experience is required for a private pilot certificate with an airplane single-engine rating?",
     "category": "Part 61 (certification)", "answerable": True, "expect_sources": [P61]},
    {"q": "What is the minimum age to be eligible for a private pilot certificate?",
     "category": "Part 61 (certification)", "answerable": True, "expect_sources": [P61]},
    {"q": "How often is a flight review required, and what must it include?",
     "category": "Part 61 (certification)", "answerable": True, "expect_sources": [P61]},
    {"q": "What recent flight experience must a pilot have to carry passengers?",
     "category": "Part 61 (certification)", "answerable": True, "expect_sources": [P61]},
    {"q": "What are the night takeoff and landing currency requirements to carry passengers at night?",
     "category": "Part 61 (certification)", "answerable": True, "expect_sources": [P61]},
    {"q": "What flight-time experience is required for a commercial pilot certificate?",
     "category": "Part 61 (certification)", "answerable": True, "expect_sources": [P61]},
    {"q": "What are the privileges and limitations of a sport pilot certificate?",
     "category": "Part 61 (certification)", "answerable": True, "expect_sources": [P61]},

    # ── Part 67 — medical standards ──────────────────────────────
    {"q": "Which medical conditions disqualify an applicant for a first-class airman medical certificate?",
     "category": "Part 67 (medical)", "answerable": True, "expect_sources": [P67]},
    # BasicMed IS in the corpus — Part 68 (§ 68.1/§ 68.3) is bundled inside the
    # Part 67 PDF, and § 61.113(i) governs the operating limits — so this is a
    # grounded, answerable question, not a no-info case.
    {"q": "What are the requirements to fly under BasicMed instead of holding an FAA medical certificate?",
     "category": "Part 67 (medical)", "answerable": True, "expect_sources": [P67, P61]},
    {"q": "What are the distant and near vision standards for a first-class medical certificate?",
     "category": "Part 67 (medical)", "answerable": True, "expect_sources": [P67]},
    {"q": "What mental-health conditions are disqualifying for an airman medical certificate?",
     "category": "Part 67 (medical)", "answerable": True, "expect_sources": [P67]},
    {"q": "What are the ear, nose, throat, and equilibrium standards for a first-class medical?",
     "category": "Part 67 (medical)", "answerable": True, "expect_sources": [P67]},

    # ── Part 71 — airspace designation ───────────────────────────
    {"q": "What are the vertical limits of Class A airspace?",
     "category": "Part 71 (airspace)", "answerable": True, "expect_sources": [P71]},
    {"q": "How is Class C airspace defined and designated?",
     "category": "Part 71 (airspace)", "answerable": True, "expect_sources": [P71]},
    {"q": "What is Class E airspace?",
     "category": "Part 71 (airspace)", "answerable": True, "expect_sources": [P71]},

    # ── Part 73 — special use airspace ───────────────────────────
    {"q": "What is a restricted area and who is the using agency?",
     "category": "Part 73 (special use)", "answerable": True, "expect_sources": [P73]},
    {"q": "What is a prohibited area?",
     "category": "Part 73 (special use)", "answerable": True, "expect_sources": [P73]},

    # ── Part 91 — general operating rules ────────────────────────
    {"q": "What are the fuel-reserve requirements for VFR flight, day versus night?",
     "category": "Part 91 (operations)", "answerable": True, "expect_sources": [P91]},
    {"q": "What are the basic VFR weather minimums?",
     "category": "Part 91 (operations)", "answerable": True, "expect_sources": [P91]},
    {"q": "When is supplemental oxygen required for the flight crew and passengers?",
     "category": "Part 91 (operations)", "answerable": True, "expect_sources": [P91]},
    {"q": "What are the right-of-way rules when two aircraft are converging?",
     "category": "Part 91 (operations)", "answerable": True, "expect_sources": [P91]},
    {"q": "What are the minimum safe altitudes over congested and non-congested areas?",
     "category": "Part 91 (operations)", "answerable": True, "expect_sources": [P91]},
    {"q": "What are the aircraft speed limits below 10,000 feet and near Class B, C, and D airspace?",
     "category": "Part 91 (operations)", "answerable": True, "expect_sources": [P91]},
    {"q": "When is an emergency locator transmitter (ELT) required and how often must its battery be replaced?",
     "category": "Part 91 (operations)", "answerable": True, "expect_sources": [P91]},
    {"q": "What preflight action is required before a flight not in the vicinity of an airport?",
     "category": "Part 91 (operations)", "answerable": True, "expect_sources": [P91]},
    {"q": "When must a transponder with altitude reporting (Mode C) be used?",
     "category": "Part 91 (operations)", "answerable": True, "expect_sources": [P91]},

    # ── Cross-section / comparison (needs 2+ sections) ───────────
    # Operating requirements live in Part 91 (§91.130 Class C, §91.131 Class B);
    # Part 71 only *designates* the airspace and has no operating rules.
    {"q": "How do operating requirements differ between Class B and Class C airspace?",
     "category": "Cross-section", "answerable": True, "expect_sources": [P91]},
    {"q": "What must a pilot do before operating in an active restricted area?",
     "category": "Cross-section", "answerable": True, "expect_sources": [P73, P91]},
    {"q": "How do the fuel-reserve requirements for VFR flight compare with those for IFR flight?",
     "category": "Cross-section", "answerable": True, "expect_sources": [P91]},

    # ── Out-of-corpus / noise (should return no-info) ────────────
    {"q": "What are the FAA requirements to register and operate a small drone (sUAS) under Part 107?",
     "category": "Out-of-corpus (aviation)", "answerable": False, "expect_sources": []},
    {"q": "What is the best recipe for tomato pasta?",
     "category": "Off-topic noise", "answerable": False, "expect_sources": []},
]


# ════════════════════════════════════════════════════════════════
# 20 questions whose answer is NOT in the corpus — negative tests.
#
# The system should refuse / return the no-info response rather than
# hallucinate. Two flavors:
#   - aviation but governed by EXCLUDED parts (Part 65/68/101/117/139, …).
#     NOTE: the corpus *cross-references* several of these part numbers
#     (e.g. "Part 121/135/141", "ultralight", "parachuting" all appear in
#     Parts 61/91), but it does NOT contain their substantive rules, so the
#     actual answer is still absent. These are the HARD cases — domain-adjacent,
#     so vector search surfaces loosely-related chunks above threshold and the
#     LLM must recognize they don't answer. (Cleanly absent — not even a
#     cross-reference: BasicMed/Part 68, ATC controller & dispatcher/Part 65,
#     flight duty period/Part 117, moored balloons/Part 101.)
#   - clearly off-topic (cooking, history, sports, …) — should be gated by
#     the similarity threshold with no LLM call at all.
#
# (BasicMed/Part 68 was previously listed here as "cleanly absent" — it is NOT:
# § 68.1/§ 68.3 are bundled in the Part 67 PDF, so that question moved to the
# answerable set above.)
#
# Run with: python eval.py --negatives   (or --all to include the set above)
# ════════════════════════════════════════════════════════════════

NO_ANSWER_QUESTIONS = [
    # Aviation, but in CFR parts NOT in the corpus (verified absent) ──
    {"q": "What certification is required to become an air traffic control tower operator?",
     "category": "No-answer (aviation: Part 65)", "answerable": False, "expect_sources": []},
    {"q": "What knowledge and experience are required for an aircraft dispatcher certificate?",
     "category": "No-answer (aviation: Part 65)", "answerable": False, "expect_sources": []},
    {"q": "What are the flight, duty, and rest period limitations for flightcrew members in scheduled operations?",
     "category": "No-answer (aviation: Part 117)", "answerable": False, "expect_sources": []},
    {"q": "What are the requirements for an airport to obtain an FAA operating certificate?",
     "category": "No-answer (aviation: Part 139)", "answerable": False, "expect_sources": []},
    {"q": "What are the operating limitations for moored balloons, kites, and unmanned free balloons?",
     "category": "No-answer (aviation: Part 101)", "answerable": False, "expect_sources": []},
    {"q": "What weight and airspeed limits define an ultralight vehicle, and where may it be operated?",
     "category": "No-answer (aviation: Part 103)", "answerable": False, "expect_sources": []},
    {"q": "What notification must a parachute jump operation give to air traffic control before jumping?",
     "category": "No-answer (aviation: Part 105)", "answerable": False, "expect_sources": []},
    {"q": "What are the dispatch and operational-control requirements for a scheduled airline under Part 121?",
     "category": "No-answer (aviation: Part 121)", "answerable": False, "expect_sources": []},
    {"q": "What duty-time limits apply to on-demand charter pilots under Part 135?",
     "category": "No-answer (aviation: Part 135)", "answerable": False, "expect_sources": []},
    {"q": "What are the requirements to obtain an FAA pilot school certificate under Part 141?",
     "category": "No-answer (aviation: Part 141)", "answerable": False, "expect_sources": []},
    {"q": "How much does a private pilot written knowledge test cost and where do I schedule it?",
     "category": "No-answer (aviation: not regulated)", "answerable": False, "expect_sources": []},

    # Clearly off-topic — should be gated by the threshold ────────────
    {"q": "What is the best recipe for a New York style cheesecake?",
     "category": "No-answer (off-topic)", "answerable": False, "expect_sources": []},
    {"q": "Who won the 2022 FIFA World Cup final?",
     "category": "No-answer (off-topic)", "answerable": False, "expect_sources": []},
    {"q": "How do I center a div horizontally and vertically in CSS?",
     "category": "No-answer (off-topic)", "answerable": False, "expect_sources": []},
    {"q": "What is the capital city of Australia?",
     "category": "No-answer (off-topic)", "answerable": False, "expect_sources": []},
    {"q": "What were the main causes of World War I?",
     "category": "No-answer (off-topic)", "answerable": False, "expect_sources": []},
    {"q": "How does compound interest work on a savings account?",
     "category": "No-answer (off-topic)", "answerable": False, "expect_sources": []},
    {"q": "What is the recommended daily water intake for a healthy adult?",
     "category": "No-answer (off-topic)", "answerable": False, "expect_sources": []},
    {"q": "Write a short haiku about autumn leaves.",
     "category": "No-answer (off-topic)", "answerable": False, "expect_sources": []},
]


# ════════════════════════════════════════════════════════════════
# Multi-turn follow-up sequences — check conversation continuity.
#
# Each sequence is a list of turns. A later turn deliberately relies on the
# earlier turn's context (an elliptical "How about ...?" / "그럼 …?"), so a
# stateless system would mis-answer or ask for clarification. A system that
# keeps history should resolve the reference and stay on the same topic.
#
# Run with: python eval.py --followups
# ════════════════════════════════════════════════════════════════

FOLLOWUP_SEQUENCES = [
    # VFR weather minimums, then shift the airspace class only.
    [
        "What are the basic VFR weather minimums for Class C airspace?",
        "How about for Class B?",
    ],
    # Private-pilot experience, then narrow to just the night requirement.
    [
        "What aeronautical experience is required for a private pilot certificate "
        "with an airplane single-engine rating?",
        "그럼 야간 비행 요건만 다시 정리해줘.",
    ],
    # First-class medical vision standard, then compare to another class.
    [
        "What are the distant vision standards for a first-class medical certificate?",
        "Are the requirements any different for a third-class medical?",
    ],
]
