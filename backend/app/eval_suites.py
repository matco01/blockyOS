"""Built-in test suites. Every question has a machine-checkable answer:
no AI grades AI. Keep answers unambiguous.

Question kinds:
  number  - final number must equal `answer` (tolerance `tol`)
  text    - final answer must contain one of `accept` (case-insensitive, whole word)
  rules   - output must pass every check in `rules` (see evals.check_rules)
  code    - extract a ```python block and run `tests` against it in a sandbox
"""

ANSWER_HINT = "\n\nThink it through if needed, then give your final answer on the last line in the form: Answer: <answer>"

SUITES = {
    "math": {
        "title": "Math", "icon": "calculator", "color": "#7c5cff",
        "about": "Everyday word problems: percentages, rates, averages.",
        "questions": [
            {"q": "A shirt costs $25 and is on sale for 20% off. What is the sale price in dollars?", "kind": "number", "answer": 20},
            {"q": "Tom has 3 boxes with 12 apples each. He gives away 15 apples. How many apples does he have left?", "kind": "number", "answer": 21},
            {"q": "A train travels at 80 km/h for 2.5 hours. How many kilometers does it travel?", "kind": "number", "answer": 200},
            {"q": "What is 17 × 24?", "kind": "number", "answer": 408},
            {"q": "What is 15% of 240?", "kind": "number", "answer": 36},
            {"q": "If 5 workers build a wall in 12 days, how many days would 10 workers take at the same rate?", "kind": "number", "answer": 6},
            {"q": "Sara reads 30 pages a day. How many days does she need to finish a 450-page book?", "kind": "number", "answer": 15},
            {"q": "The average of 4, 8, 15 and x is 10. What is x?", "kind": "number", "answer": 13},
            {"q": "A number is doubled and then increased by 7, giving 31. What is the number?", "kind": "number", "answer": 12},
            {"q": "How many minutes are there in 3.5 hours?", "kind": "number", "answer": 210},
            {"q": "Pencils cost 3 for $1. How many dollars do 24 pencils cost?", "kind": "number", "answer": 8},
            {"q": "A tank holds 120 liters and is 35% full. How many more liters are needed to fill it?", "kind": "number", "answer": 78},
        ],
    },
    "logic": {
        "title": "Logic", "icon": "puzzle", "color": "#ff5c9d",
        "about": "Classic trick questions that catch models answering on autopilot.",
        "questions": [
            {"q": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost, in cents?", "kind": "number", "answer": 5},
            {"q": "If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Answer yes or no.", "kind": "text", "accept": ["yes"]},
            {"q": "Mary's father has five daughters: Nana, Nene, Nini and Nono. What is the name of the fifth daughter?", "kind": "text", "accept": ["mary"]},
            {"q": "A farmer has 17 sheep. All but 9 run away. How many sheep are left?", "kind": "number", "answer": 9},
            {"q": "Which is heavier: a kilogram of feathers or a kilogram of steel? Answer 'feathers', 'steel' or 'same'.", "kind": "text", "accept": ["same", "neither", "equal"]},
            {"q": "In a race, you overtake the person in second place. What place are you in now?", "kind": "text", "accept": ["second", "2nd", "2"]},
            {"q": "How many times does the letter 'r' appear in the word 'strawberry'?", "kind": "number", "answer": 3},
            {"q": "If 5 machines take 5 minutes to make 5 widgets, how many minutes would 100 machines take to make 100 widgets?", "kind": "number", "answer": 5},
            {"q": "A patch of lily pads doubles in size every day. It takes 48 days to cover the whole lake. How many days does it take to cover half the lake?", "kind": "number", "answer": 47},
            {"q": "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?", "kind": "text", "accept": ["carol"]},
            {"q": "What comes next in the sequence 2, 6, 12, 20, 30, ...?", "kind": "number", "answer": 42},
            {"q": "Today is Wednesday. What day of the week will it be 10 days from today?", "kind": "text", "accept": ["saturday"]},
        ],
    },
    "hard": {
        "title": "Hard reasoning", "icon": "brain", "color": "#e879f9",
        "about": "Multi-step problems and known traps that separate strong models from weak ones.",
        "questions": [
            {"q": "A store raises a price by 25% and then lowers the new price by 20%. By what percent has the price changed overall? (Use 0 if unchanged.)", "kind": "number", "answer": 0},
            {"q": "How many positive integers less than 100 are divisible by 3 or by 5?", "kind": "number", "answer": 46},
            {"q": "What is the sum of all integers from 1 to 200 that are NOT divisible by 4?", "kind": "number", "answer": 15000},
            {"q": "A clock shows 3:15. What is the smaller angle between the hour hand and the minute hand, in degrees?", "kind": "number", "answer": 7.5, "tol": 0.01},
            {"q": "In a family, each boy has as many sisters as brothers, but each girl has twice as many brothers as sisters. How many children are in the family?", "kind": "number", "answer": 7},
            {"q": "How many times does the letter 's' appear (upper or lower case) in the sentence 'She sells seashells by the seashore'?", "kind": "number", "answer": 8},
            {"q": "What is the 10th prime number?", "kind": "number", "answer": 29},
            {"q": "If you write down every whole number from 1 to 100, how many times do you write the digit 9?", "kind": "number", "answer": 20},
            {"q": "A snail is at the bottom of a 10-meter well. Each day it climbs 3 meters and each night it slides back 2 meters. On which day does it first reach the top?", "kind": "number", "answer": 8},
            {"q": "Three friends split a $90 bill. One pays twice as much as each of the other two, who pay equal amounts. How many dollars does the biggest payer pay?", "kind": "number", "answer": 45},
            {"q": "Which number is larger: 9.11 or 9.9?", "kind": "number", "answer": 9.9, "tol": 0.001},
            {"q": "How many times does the letter 'r' appear in the word 'refrigerator'?", "kind": "number", "answer": 4},
        ],
    },
    "knowledge": {
        "title": "Knowledge", "icon": "book", "color": "#ffa94d",
        "about": "General facts. Small models often know less, not reason worse.",
        "questions": [
            {"q": "What is the capital city of Australia?", "kind": "text", "accept": ["canberra"]},
            {"q": "What is the chemical symbol for gold?", "kind": "text", "accept": ["au"]},
            {"q": "Who wrote the novel 'Pride and Prejudice'?", "kind": "text", "accept": ["austen"]},
            {"q": "What is the largest planet in our solar system?", "kind": "text", "accept": ["jupiter"]},
            {"q": "At sea level, at what temperature in degrees Celsius does water boil?", "kind": "number", "answer": 100},
            {"q": "How many bones are in the adult human body?", "kind": "number", "answer": 206},
            {"q": "In what year did the Berlin Wall fall?", "kind": "number", "answer": 1989},
            {"q": "Which element has atomic number 6?", "kind": "text", "accept": ["carbon"]},
            {"q": "Which part of a cell produces most of its ATP?", "kind": "text", "accept": ["mitochondria", "mitochondrion"]},
            {"q": "In which country is the city of Kyoto?", "kind": "text", "accept": ["japan"]},
            {"q": "Who painted the Mona Lisa?", "kind": "text", "accept": ["leonardo", "da vinci"]},
            {"q": "What is the smallest prime number?", "kind": "number", "answer": 2},
        ],
    },
    "instructions": {
        "title": "Instructions", "icon": "list-checks", "color": "#22d3ee",
        "about": "Does it follow exact formatting rules? Checked by code, not opinion.",
        "questions": [
            {"q": "List exactly three fruits as a bulleted list. Output nothing else.", "kind": "rules", "rules": [["bullets", 3], ["only_list"]]},
            {"q": "Write one sentence about the ocean using only lowercase letters.", "kind": "rules", "rules": [["no_upper"], ["min_words", 4]]},
            {"q": "Describe a cat in fewer than 15 words.", "kind": "rules", "rules": [["max_words", 14], ["min_words", 3]]},
            {"q": "Reply with only the word banana, nothing else.", "kind": "rules", "rules": [["exactly", "banana"]]},
            {"q": "Write one sentence about winter that includes the words 'snow' and 'quiet'.", "kind": "rules", "rules": [["contains", "snow"], ["contains", "quiet"]]},
            {"q": "Return a JSON object with the keys \"name\" and \"age\" for a fictional person. Output only the JSON.", "kind": "rules", "rules": [["json_keys", ["name", "age"]]]},
            {"q": "Write a short poem about rain with exactly 3 lines.", "kind": "rules", "rules": [["lines", 3]]},
            {"q": "Answer in ALL CAPITAL LETTERS: what color is the sky on a clear day?", "kind": "rules", "rules": [["all_caps"], ["contains", "blue"]]},
            {"q": "Write two sentences about coffee without using any commas.", "kind": "rules", "rules": [["no_char", ","], ["contains", "coffee"]]},
            {"q": "Give a numbered list of 5 countries in Europe. Output nothing else.", "kind": "rules", "rules": [["numbered", 5], ["only_list"]]},
            {"q": "Start your answer with the word 'Certainly' and explain what a battery does in one sentence.", "kind": "rules", "rules": [["starts_with", "certainly"]]},
            {"q": "Write exactly 4 words about the moon.", "kind": "rules", "rules": [["word_count", 4]]},
        ],
    },
    "coding": {
        "title": "Coding", "icon": "code", "color": "#34d399",
        "about": "Writes Python functions; we run them against unit tests in a locked sandbox.",
        "questions": [
            {"q": "Write a Python function `is_palindrome(s)` that returns True if the string reads the same forwards and backwards, ignoring case and any non-alphanumeric characters.",
             "kind": "code", "tests": ["assert is_palindrome('A man, a plan, a canal: Panama')", "assert not is_palindrome('hello')", "assert is_palindrome('')", "assert is_palindrome('No lemon, no melon')"]},
            {"q": "Write a Python function `fizzbuzz(n)` that returns a list of strings for 1..n: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, otherwise the number as a string.",
             "kind": "code", "tests": ["assert fizzbuzz(5) == ['1','2','Fizz','4','Buzz']", "assert fizzbuzz(15)[-1] == 'FizzBuzz'", "assert len(fizzbuzz(30)) == 30"]},
            {"q": "Write a Python function `second_largest(nums)` that returns the second largest distinct number in a list, or None if there isn't one.",
             "kind": "code", "tests": ["assert second_largest([3, 1, 4, 4, 2]) == 3", "assert second_largest([5]) is None", "assert second_largest([7, 7]) is None", "assert second_largest([-1, -5, -3]) == -3"]},
            {"q": "Write a Python function `count_vowels(s)` that returns how many vowels (a, e, i, o, u, case-insensitive) are in the string.",
             "kind": "code", "tests": ["assert count_vowels('Hello World') == 3", "assert count_vowels('') == 0", "assert count_vowels('AEIOU xyz') == 5"]},
            {"q": "Write a Python function `flatten(items)` that flattens an arbitrarily nested list into a flat list.",
             "kind": "code", "tests": ["assert flatten([1, [2, [3, [4]], 5]]) == [1, 2, 3, 4, 5]", "assert flatten([]) == []", "assert flatten([[[]]]) == []"]},
            {"q": "Write a Python function `fib(n)` that returns the n-th Fibonacci number, with fib(0) = 0 and fib(1) = 1. It must be fast for n = 90.",
             "kind": "code", "tests": ["assert fib(0) == 0", "assert fib(1) == 1", "assert fib(10) == 55", "assert fib(90) == 2880067194370816120"]},
            {"q": "Write a Python function `is_prime(n)` that returns True if n is a prime number.",
             "kind": "code", "tests": ["assert is_prime(2) and is_prime(97)", "assert not is_prime(1) and not is_prime(0) and not is_prime(91)", "assert is_prime(7919)"]},
            {"q": "Write a Python function `word_counts(text)` that returns a dict mapping each lowercase word to how often it appears. Words are separated by whitespace; strip the punctuation .,!? from their ends.",
             "kind": "code", "tests": ["assert word_counts('The cat. The dog!') == {'the': 2, 'cat': 1, 'dog': 1}", "assert word_counts('') == {}"]},
        ],
    },
}

CODE_HINT = "\n\nReply with only the complete function in a single ```python code block. No explanations, no example usage."
