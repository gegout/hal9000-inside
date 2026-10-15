import json

def validate_turns(name, turns, is_alternating=False):
    assert len(turns) >= 50, f"{name}: expected >= 50 turns, got {len(turns)}"
    for idx, t in enumerate(turns, 1):
        wc = len(t.split())
        if not is_alternating:
            assert wc < 20, f"{name} turn {idx} has {wc} words (must be < 20): {t}"
        else:
            if idx % 2 == 1:
                assert wc < 10, f"{name} odd turn {idx} has {wc} words (must be < 10): {t}"
            else:
                assert 25 <= wc <= 35, f"{name} even turn {idx} has {wc} words (must be 25-35): {t}"
    print(f"PASS: {name} (50 turns, all constraints satisfied)")

