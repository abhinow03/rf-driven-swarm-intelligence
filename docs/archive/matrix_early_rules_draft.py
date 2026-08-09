from itertools import product

FORMATIONS = ["v_shape", "encirclement", "column", "diamond", "dispersed", "converging", "shield"]

RULES = {
    # steady-state
    ("v_shape",      "v_shape"):      ("medium",   "surveillance",        "increase_surveillance"),
    ("encirclement", "encirclement"): ("high",     "encircle",            "alert_operator"),
    ("column",       "column"):       ("low",      "patrol",              "monitor"),
    ("diamond",      "diamond"):      ("low",      "patrol",              "monitor"),
    ("dispersed",    "dispersed"):    ("low",      "surveillance",        "monitor"),
    ("converging",   "converging"):   ("high",     "approach",            "alert_operator"),
    ("shield",       "shield"):       ("medium",   "defensive",           "monitor"),
    # to converging
    ("v_shape",      "converging"):   ("high",     "approach",            "alert_operator"),
    ("encirclement", "converging"):   ("critical", "encircle",            "deploy_countermeasure"),
    ("column",       "converging"):   ("high",     "approach",            "alert_operator"),
    ("diamond",      "converging"):   ("high",     "approach",            "alert_operator"),
    ("dispersed",    "converging"):   ("high",     "approach",            "alert_operator"),
    ("shield",       "converging"):   ("medium",   "approach",            "increase_surveillance"),
    # to encirclement
    ("v_shape",      "encirclement"): ("high",     "encircle",            "alert_operator"),
    ("column",       "encirclement"): ("high",     "encircle",            "alert_operator"),
    ("diamond",      "encirclement"): ("high",     "encircle",            "alert_operator"),
    ("dispersed",    "encirclement"): ("high",     "encircle",            "alert_operator"),
    ("converging",   "encirclement"): ("critical", "encircle",            "deploy_countermeasure"),
    ("shield",       "encirclement"): ("medium",   "encircle",            "increase_surveillance"),
    # from encirclement
    ("encirclement", "v_shape"):      ("medium",   "regroup",             "increase_surveillance"),
    ("encirclement", "column"):       ("low",      "withdraw",            "monitor"),
    ("encirclement", "diamond"):      ("medium",   "consolidate",         "monitor"),
    ("encirclement", "dispersed"):    ("low",      "withdraw",            "monitor"),
    ("encirclement", "shield"):       ("medium",   "defensive",           "monitor"),
    # from converging
    ("converging",   "v_shape"):      ("medium",   "regroup",             "increase_surveillance"),
    ("converging",   "column"):       ("low",      "patrol",              "monitor"),
    ("converging",   "diamond"):      ("medium",   "consolidate",         "monitor"),
    ("converging",   "dispersed"):    ("low",      "withdraw",            "monitor"),
    ("converging",   "shield"):       ("medium",   "defensive",           "monitor"),
    # v_shape remaining
    ("v_shape",      "column"):       ("low",      "transit",             "monitor"),
    ("v_shape",      "diamond"):      ("medium",   "defensive_transition","monitor"),
    ("v_shape",      "dispersed"):    ("low",      "area_search",         "monitor"),
    ("v_shape",      "shield"):       ("medium",   "defensive_transition","monitor"),
    # column remaining
    ("column",       "v_shape"):      ("high",     "attack_preparation",  "alert_operator"),
    ("column",       "diamond"):      ("medium",   "consolidate",         "monitor"),
    ("column",       "dispersed"):    ("low",      "area_search",         "monitor"),
    ("column",       "shield"):       ("medium",   "defensive_transition","monitor"),
    # diamond remaining
    ("diamond",      "v_shape"):      ("medium",   "reposition",          "increase_surveillance"),
    ("diamond",      "column"):       ("low",      "transit",             "monitor"),
    ("diamond",      "dispersed"):    ("medium",   "area_search",         "monitor"),
    ("diamond",      "shield"):       ("medium",   "defensive_transition","monitor"),
    # dispersed remaining
    ("dispersed",    "v_shape"):      ("high",     "rally",               "alert_operator"),
    ("dispersed",    "column"):       ("low",      "transit",             "monitor"),
    ("dispersed",    "diamond"):      ("medium",   "consolidate",         "increase_surveillance"),
    ("dispersed",    "shield"):       ("medium",   "defensive_transition","monitor"),
    # shield remaining
    ("shield",       "v_shape"):      ("medium",   "reposition",          "increase_surveillance"),
    ("shield",       "column"):       ("medium",   "reposition",          "monitor"),
    ("shield",       "diamond"):      ("medium",   "defensive",           "monitor"),
    ("shield",       "dispersed"):    ("low",      "surveillance",        "monitor"),
}

DEFAULT = ("medium", "reposition", "monitor")

counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
print(f"{'from->to':<30} {'threat':<10} {'intent':<20} {'action':<30} explicit?")
print("-" * 95)
for a, b in product(FORMATIONS, FORMATIONS):
    rule = RULES.get((a, b), DEFAULT)
    explicit = (a, b) in RULES
    counts[rule[0]] += 1
    print(f"{a+' -> '+b:<30} {rule[0]:<10} {rule[1]:<20} {rule[2]:<30} {'YES' if explicit else 'DEFAULT'}")

print()
print("Threat distribution:", counts)
gaps = [(a, b) for a, b in product(FORMATIONS, FORMATIONS) if (a, b) not in RULES]
print("Pairs falling to DEFAULT:", gaps or "None")
