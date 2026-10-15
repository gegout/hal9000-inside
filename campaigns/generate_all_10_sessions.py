import json

def verify_and_save():
    from campaigns.build_all_scenarios import get_session_1, get_session_2, get_session_3, get_session_4, get_session_5
    
    s1 = get_session_1()
    s2 = get_session_2()
    s3 = get_session_3()
    s4 = get_session_4()
    s5 = get_session_5()

    for idx, s in enumerate([s1, s2, s3, s4, s5], 1):
        assert len(s) >= 50, f"Session {idx} has {len(s)} turns (need >= 50)"
        for t_idx, t in enumerate(s, 1):
            wc = len(t.split())
            assert wc < 20, f"Session {idx} turn {t_idx} has {wc} words (must be < 20): {t}"

    print("Sessions 1 to 5 passed all constraints!")

verify_and_save()
