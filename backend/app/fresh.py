"""Fresh tests: brand-new questions generated at the moment a run starts.

A model can't have memorized a question that didn't exist a second ago, so
these measure skill, not recall (the idea behind GSM-Symbolic). Every
question is built from a template with random names and numbers; the answer
is computed by code. The run stores its seed, so a run can be reproduced.
"""
import random
import string

NAMES = ["Ava", "Liam", "Noor", "Mateo", "Yuki", "Zara", "Omar", "Ines", "Kofi", "Lena", "Ravi", "Mila",
         "Tomas", "Aisha", "Jonas", "Priya", "Felix", "Hana", "Diego", "Sofia", "Elio", "Nadia", "Bram", "Iris"]
ITEMS = [("notebook", "notebooks"), ("apple", "apples"), ("ticket", "tickets"), ("candle", "candles"),
         ("pencil", "pencils"), ("mug", "mugs"), ("plant", "plants"), ("battery", "batteries"), ("card", "cards")]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _names(r: random.Random, n: int) -> list[str]:
    return r.sample(NAMES, n)


# ---------------- Math (number answers) ----------------

def m_change(r):
    n = r.choice(NAMES); item, items = r.choice(ITEMS)
    a, p = r.randint(3, 12), r.randint(2, 19)
    bill = next(b for b in (20, 50, 100, 200, 500) if b > a * p)  # max cost is 12 × 19 = 228
    return f"{n} buys {a} {items} at ${p} each and pays with a ${bill} bill. How many dollars of change does {n} get?", bill - a * p


def m_discount(r):
    price, pct = r.choice([40, 60, 80, 120, 150, 240, 360]), r.choice([10, 15, 20, 25, 30, 40])
    extra = r.choice([0, 5, 10])
    final = price * (1 - pct / 100) * (1 - extra / 100)
    tail = f" Then an extra {extra}% is taken off the reduced price." if extra else ""
    return f"A jacket costs ${price}. It is discounted by {pct}%.{tail} What is the final price in dollars?", round(final, 2)


def m_distance(r):
    s1, h1, s2, h2 = r.randint(40, 90), r.randint(1, 4), r.randint(20, 70), r.choice([0.5, 1.5, 2, 2.5])
    return f"A car drives at {s1} km/h for {h1} hours, then at {s2} km/h for {h2} hours. How many kilometers does it travel in total?", s1 * h1 + s2 * h2


def m_workers(r):
    w1, w2 = r.sample([2, 3, 4, 5, 6, 8, 10, 12], 2)
    d1 = w2 * r.randint(1, 6)  # makes w1 * d1 divisible by w2, so the answer is a whole number
    return f"{w1} workers can paint a building in {d1} days. How many days would {w2} workers take, working at the same rate?", w1 * d1 // w2


def m_average(r):
    k = r.randint(3, 5); vals = [r.randint(5, 60) for _ in range(k)]
    x = r.randint(5, 60)
    x += (-(sum(vals) + x)) % (k + 1)  # nudge x so the average is a whole number
    target = (sum(vals) + x) // (k + 1)
    return f"The average of {', '.join(map(str, vals))} and x is {target}. What is x?", x


def m_ratio(r):
    a, b = r.randint(1, 6), r.randint(1, 6)
    while a == b:
        b = r.randint(1, 7)
    unit = r.randint(4, 30); total = unit * (a + b)
    n1, n2 = _names(r, 2)
    return f"{n1} and {n2} share ${total} in the ratio {a}:{b}. How many dollars does {n1 if a > b else n2} (the larger share) get?", unit * max(a, b)


def m_ages(r):
    n1, n2 = _names(r, 2)
    b = r.randint(5, 30); diff = r.randint(2, 15); later = r.randint(2, 10)
    s = (b + diff + later) + (b + later)
    return f"{n1} is {diff} years older than {n2}. In {later} years, the sum of their ages will be {s}. How old is {n2} now?", b


def m_tank(r):
    cap = r.choice([200, 240, 300, 360, 400, 480]); pct = r.choice([15, 20, 25, 35, 40, 60, 75])
    used = r.randint(5, cap * pct // 100 - 1) if cap * pct // 100 > 6 else 0
    have = cap * pct / 100 - used
    return (f"A tank holds {cap} liters and is {pct}% full. Then {used} liters are used. How many liters must be added to fill it completely?",
            round(cap - have, 2))


def m_multi(r):
    n = r.choice(NAMES); item, items = r.choice(ITEMS)
    boxes, per, gave, bought = r.randint(3, 9), r.randint(6, 15), r.randint(5, 20), r.randint(2, 12)
    return (f"{n} has {boxes} boxes with {per} {items} in each. {n} gives away {gave} {items} and then buys {bought} more. "
            f"How many {items} does {n} have now?", boxes * per - gave + bought)


def m_minutes(r):
    h, m = r.randint(1, 5), r.choice([15, 20, 30, 45])
    rate = r.choice([2, 3, 4, 5])
    return f"A machine makes {rate} parts per minute. How many parts does it make in {h} hours and {m} minutes?", (h * 60 + m) * rate


MATH = [m_change, m_discount, m_distance, m_workers, m_average, m_ratio, m_ages, m_tank, m_multi, m_minutes]


# ---------------- Logic (short answers) ----------------

def _pseudo_word(r, lo=7, hi=12):
    cons, vow = "bcdfghjklmnprstvz", "aeiou"
    return "".join(r.choice(cons) if i % 2 == 0 else r.choice(vow) for i in range(r.randint(lo, hi)))


def l_order(r):
    k = r.randint(4, 6); people = _names(r, k)  # tallest -> shortest
    facts = [f"{people[i]} is taller than {people[i + 1]}." for i in range(k - 1)]
    r.shuffle(facts)
    ask_short = r.random() < 0.5
    return {"q": " ".join(facts) + f" Who is the {'shortest' if ask_short else 'tallest'}?", "kind": "text",
            "accept": [(people[-1] if ask_short else people[0]).lower()]}


def l_calendar(r):
    d = r.randrange(7); n = r.randint(3, 400)
    return {"q": f"Today is {DAYS[d]}. What day of the week will it be {n} days from today?", "kind": "text",
            "accept": [DAYS[(d + n) % 7].lower()]}


def l_letters(r):
    word = _pseudo_word(r, 9, 14)
    ch = r.choice(sorted(set(word)))
    return {"q": f"How many times does the letter '{ch}' appear in the made-up word '{word}'?", "kind": "number", "answer": word.count(ch)}


def l_reverse(r):
    word = _pseudo_word(r, 7, 10)
    return {"q": f"Write the made-up word '{word}' backwards, letter by letter.", "kind": "text", "accept": [word[::-1]]}


def l_sequence(r):
    t = r.choice(["arith", "geo", "quad"])
    if t == "arith":
        a, d = r.randint(-20, 40), r.choice([3, 4, 6, 7, 9, 11, -5])
        seq = [a + i * d for i in range(6)]
    elif t == "geo":
        a, q = r.randint(1, 5), r.choice([2, 3])
        seq = [a * q ** i for i in range(6)]
    else:
        a, b, c = r.randint(1, 3), r.randint(-3, 5), r.randint(0, 9)
        seq = [a * i * i + b * i + c for i in range(1, 7)]
    return {"q": f"What is the next number in the sequence {', '.join(map(str, seq[:5]))}, ...?", "kind": "number", "answer": seq[5]}


def l_divisible(r):
    n = r.choice([100, 150, 200, 300, 500]); a, b = r.sample([2, 3, 4, 5, 6, 7, 9, 11], 2)
    count = sum(1 for i in range(1, n) if i % a == 0 or i % b == 0)
    return {"q": f"How many positive integers less than {n} are divisible by {a} or by {b}?", "kind": "number", "answer": count}


def l_decimal(r):
    whole = r.randint(1, 20); x = f"{whole}.{r.randint(10, 19)}"; y = f"{whole}.{r.randint(2, 9)}"
    pair = [x, y]; r.shuffle(pair)
    return {"q": f"Which number is larger: {pair[0]} or {pair[1]}?", "kind": "number", "answer": max(float(x), float(y))}


def l_syllogism(r):
    a, b, c = (_pseudo_word(r, 4, 6) + "s" for _ in range(3))
    valid = r.random() < 0.5
    if valid:
        q = f"All {a} are {b}. All {b} are {c}. Are all {a} definitely {c}? Answer yes or no."
    else:
        q = f"All {a} are {b}. Some {b} are {c}. Are all {a} definitely {c}? Answer yes or no."
    return {"q": q, "kind": "text", "accept": ["yes" if valid else "no"]}


def l_clock(r):
    h, m = r.randint(1, 11), r.choice([0, 10, 15, 20, 30, 40, 45, 50])
    ang = abs(30 * h + 0.5 * m - 6 * m) % 360
    ang = min(ang, 360 - ang)
    return {"q": f"A clock shows {h}:{m:02d}. What is the smaller angle between the hour and minute hands, in degrees?", "kind": "number", "answer": ang, "tol": 0.01}


LOGIC = [l_order, l_calendar, l_letters, l_reverse, l_sequence, l_divisible, l_decimal, l_syllogism, l_clock]


# ---------------- Coding (unit tests generated from a reference solution) ----------------

def _fn(r):
    return "f_" + "".join(r.choice(string.ascii_lowercase) for _ in range(5))


def c_every_kth(r):
    name, k = _fn(r), r.randint(2, 4)
    spec = f"returns the sum of the numbers in the list `nums` whose index is divisible by {k} (index 0 counts)"
    ref = lambda nums: sum(v for i, v in enumerate(nums) if i % k == 0)  # noqa: E731
    cases = [[r.randint(-9, 20) for _ in range(r.randint(0, 9))] for _ in range(5)]
    return name, "nums", spec, ref, [(c,) for c in cases]


def c_long_words(r):
    name, n = _fn(r), r.randint(3, 6)
    spec = f"returns how many words in the string `text` (split on whitespace) are longer than {n} characters"
    ref = lambda text: sum(1 for w in text.split() if len(w) > n)  # noqa: E731
    cases = [" ".join(_pseudo_word(r, 2, 9) for _ in range(r.randint(0, 7))) for _ in range(5)]
    return name, "text", spec, ref, [(c,) for c in cases]


def c_rotate(r):
    name, k = _fn(r), r.randint(1, 5)
    spec = f"returns the list `items` rotated to the right by {k} positions (an empty list stays empty)"
    def ref(items):
        if not items:
            return []
        s = k % len(items)
        return items[-s:] + items[:-s] if s else list(items)
    cases = [[r.randint(0, 9) for _ in range(r.randint(0, 7))] for _ in range(5)]
    return name, "items", spec, ref, [(c,) for c in cases]


def c_replace_vowels(r):
    name, ch = _fn(r), r.choice("*#@_")
    spec = f"returns the string `s` with every vowel (a, e, i, o, u, upper or lower case) replaced by '{ch}'"
    ref = lambda s: "".join(ch if c.lower() in "aeiou" else c for c in s)  # noqa: E731
    cases = [_pseudo_word(r, 3, 10).capitalize() + " " + _pseudo_word(r, 2, 6).upper() for _ in range(5)]
    return name, "s", spec, ref, [(c,) for c in cases]


def c_range_div(r):
    name, d = _fn(r), r.randint(3, 8)
    spec = f"returns a list of all integers from `a` to `b` inclusive that are divisible by {d}, in increasing order"
    ref = lambda a, b: [x for x in range(a, b + 1) if x % d == 0]  # noqa: E731
    cases = []
    for _ in range(5):
        a = r.randint(-20, 30)
        cases.append((a, a + r.randint(0, 40)))
    return name, "a, b", spec, ref, cases


def c_caesar(r):
    name, k = _fn(r), r.randint(1, 25)
    spec = f"shifts every lowercase letter in `s` forward by {k} places in the alphabet (wrapping from z to a) and leaves all other characters unchanged"
    ref = lambda s: "".join(chr((ord(c) - 97 + k) % 26 + 97) if "a" <= c <= "z" else c for c in s)  # noqa: E731
    cases = [_pseudo_word(r, 3, 9) + r.choice([" ", "!", " Z", "-"]) + _pseudo_word(r, 2, 5) for _ in range(5)]
    return name, "s", spec, ref, [(c,) for c in cases]


def c_rle(r):
    name = _fn(r)
    spec = "run-length encodes the string `s`: each run of the same character becomes the character followed by the run length, e.g. 'aaab' -> 'a3b1' (an empty string gives '')"
    def ref(s):
        out, i = [], 0
        while i < len(s):
            j = i
            while j < len(s) and s[j] == s[i]:
                j += 1
            out.append(f"{s[i]}{j - i}")
            i = j
        return "".join(out)
    cases = ["".join(r.choice("abc") * r.randint(1, 4) for _ in range(r.randint(0, 5))) for _ in range(5)]
    return name, "s", spec, ref, [(c,) for c in cases]


def c_kth_largest(r):
    name, k = _fn(r), r.randint(2, 3)
    spec = f"returns the {k}{'nd' if k == 2 else 'rd'} largest distinct value in the list `nums`, or None if there are fewer than {k} distinct values"
    def ref(nums):
        d = sorted(set(nums), reverse=True)
        return d[k - 1] if len(d) >= k else None
    cases = [[r.randint(-5, 9) for _ in range(r.randint(0, 8))] for _ in range(5)] + [[4, 4]]
    return name, "nums", spec, ref, [(c,) for c in cases]


CODE = [c_every_kth, c_long_words, c_rotate, c_replace_vowels, c_range_div, c_caesar, c_rle, c_kth_largest]


def _code_question(r, gen) -> dict:
    name, params, spec, ref, cases = gen(r)
    tests = [f"assert {name}({', '.join(map(repr, args))}) == {ref(*args)!r}" for args in cases]
    return {"q": f"Write a Python function `{name}({params})` that {spec}.", "kind": "code", "tests": tests}


SETS = {
    "math": {"title": "Fresh math", "about": "New word problems every run: same skills as GSM8K, but numbers and names never seen before.", "color": "#7c5cff", "icon": "calculator"},
    "logic": {"title": "Fresh logic", "about": "New puzzles every run: ordering, calendars, sequences, letter counting in made-up words.", "color": "#ff5c9d", "icon": "puzzle"},
    "code": {"title": "Fresh coding", "about": "New small Python tasks every run, with tests generated from a reference solution.", "color": "#34d399", "icon": "code"},
}


def generate(kind: str, n: int, seed: int) -> list[dict]:
    """n questions of a kind, deterministic for a given seed (so a run can be reproduced)."""
    r = random.Random(f"{kind}-{seed}")
    out = []
    for i in range(n):
        if kind == "math":
            q, a = MATH[i % len(MATH)](r)
            out.append({"q": q, "kind": "number", "answer": a, "tol": 0.011, "tag": MATH[i % len(MATH)].__name__[2:]})
        elif kind == "logic":
            gen = LOGIC[i % len(LOGIC)]
            out.append({**gen(r), "tag": gen.__name__[2:]})
        elif kind == "code":
            gen = CODE[i % len(CODE)]
            out.append({**_code_question(r, gen), "tag": gen.__name__[2:]})
    r.shuffle(out)
    return out
