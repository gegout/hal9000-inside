# Script to generate exact turns for Sessions 8, 9, 10
# Rules:
# Turn index 0, 2, 4... (odd turn numbers 1, 3, 5...): strictly < 10 words
# Turn index 1, 3, 5... (even turn numbers 2, 4, 6...): strictly 25-35 words (inclusive)
# Total turns >= 50
# At least two research requests ("Search online for...")
# References to earlier turns, corrections, coherent follow-ups

import sys

def validate_turns(name, turns):
    assert len(turns) >= 50, f"{name}: length {len(turns)} < 50"
    research_count = 0
    has_ref = False
    has_corr = False
    for i, t in enumerate(turns):
        w = len(t.split())
        turn_num = i + 1
        if turn_num % 2 == 1:
            # Short turn: < 10 words
            assert w < 10, f"{name} turn {turn_num} ({w} words): '{t}' not < 10"
        else:
            # Long turn: 25-35 words
            assert 25 <= w <= 35, f"{name} turn {turn_num} ({w} words): '{t}' not between 25 and 35"
        if "search" in t.lower():
            research_count += 1
        if any(k in t.lower() for k in ["earlier", "remember", "before", "previously"]):
            has_ref = True
        if any(k in t.lower() for k in ["wait", "correct", "clarify", "actually", "mistake"]):
            has_corr = True
    assert research_count >= 2, f"{name}: only {research_count} research requests"
    assert has_ref, f"{name}: missing reference to earlier turns"
    assert has_corr, f"{name}: missing correction"
    print(f"PASS: {name} ({len(turns)} turns, {research_count} research requests)")

