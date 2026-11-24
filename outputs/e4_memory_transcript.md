# E4 - Multi-turn conversation with short-term memory

Memory design: last N turns per session (query, standalone query, answer, entities, cited docs). Follow-ups are detected (anaphora / leading 'and' / no own entity) and rewritten by carrying over the previous entities; documents cited in the previous turn get a retrieval boost. A SQLite answer cache (long-term memory) returns repeated questions with zero LLM tokens.

## with memory

**Turn 1 - user:** What eCPRI one-way latency was measured in the fronthaul tests of Site-754?
- standalone query: `What eCPRI one-way latency was measured in the fronthaul tests of Site-754?`
- answer: Measured for FH-754-01 is 76 us [1].
- sources: fh_test_site754.pptx p.3 (table)
- correct: yes

**Turn 2 - user:** Which VLAN carries its fronthaul traffic in the topology diagram?
- standalone query: `Which VLAN carries its fronthaul traffic in the topology diagram (regarding Site-754)?`
- answer: Fronthaul VLAN 2754 NG 25GE [1].
- sources: fh_test_site754.pptx p.4 (image)
- correct: yes

**Turn 3 - user:** And what software build does the O-DU there run?
- standalone query: `And what software build does the O-DU there run (regarding Site-754)?`
- answer: The O-DU at Site-754 runs software build R24.2.8, integrated with O-RU firmware FW-6.7.28 [1].
- sources: fh_test_site754.pptx p.2 (text)
- correct: yes

## without memory

**Turn 1 - user:** What eCPRI one-way latency was measured in the fronthaul tests of Site-754?
- standalone query: `What eCPRI one-way latency was measured in the fronthaul tests of Site-754?`
- answer: Measured for FH-754-01 is 76 us [1].
- sources: fh_test_site754.pptx p.3 (table)
- correct: yes

**Turn 2 - user:** Which VLAN carries its fronthaul traffic in the topology diagram?
- standalone query: `Which VLAN carries its fronthaul traffic in the topology diagram?`
- answer: Fronthaul VLAN 2106 NG 25GE [1].
- sources: fh_test_site106.pptx p.4 (image)
- correct: no

**Turn 3 - user:** And what software build does the O-DU there run?
- standalone query: `And what software build does the O-DU there run?`
- answer: O-DU Software Release Note - Build R25.3.473 Document date: 2025-02-27 Introduction This release note describes software build R25.3.473 for the O-DU [2].
- sources: release_note_473.pdf p.1 (text)
- correct: no
