"""Fixed evaluation question set for the 14 CFR (FAA regulations) RAG chatbot.

The five practice questions, each tagged with the CFR Part(s) that should
ground the answer. Keep this list stable so before/after comparisons are fair.
`expect_sources` lists the index source filenames whose content should appear
in good citations.
"""

QUESTIONS = [
    {
        "q": "What aeronautical experience is required for a private pilot "
             "certificate with an airplane single-engine rating?",
        "type": "Part 61 — single-part, answerable",
        "expect_sources": ["CFR-2025-title14-vol2-part61.pdf"],
    },
    {
        "q": "Which medical conditions disqualify an applicant for a "
             "first-class airman medical certificate?",
        "type": "Part 67 — single-part, answerable",
        "expect_sources": ["CFR-2025-title14-vol2-part67.pdf"],
    },
    {
        "q": "What are the fuel-reserve requirements for VFR flight, day "
             "versus night?",
        "type": "Part 91 — single-part, comparison",
        "expect_sources": ["CFR-2025-title14-vol2-part91.pdf"],
    },
    {
        "q": "How do operating requirements differ between Class B and "
             "Class C airspace?",
        "type": "Parts 71 + 91 — cross-part, comparison",
        "expect_sources": [
            "CFR-2025-title14-vol2-part71.pdf",
            "CFR-2025-title14-vol2-part91.pdf",
        ],
    },
    {
        "q": "What must a pilot do before operating in an active restricted "
             "area?",
        "type": "Part 73 (+ §91.133 operating rule) — cross-part",
        "expect_sources": [
            "CFR-2025-title14-vol2-part73.pdf",
            "CFR-2025-title14-vol2-part91.pdf",
        ],
    },
]
