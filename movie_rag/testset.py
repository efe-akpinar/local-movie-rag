# ruff: noqa: E501 (queries and reference answers are copied verbatim from the notebook)
"""The notebook's 20-query test set, copied verbatim (Section 7).

Categories: standard, adversarial and sequel_confusion queries have gold titles; unanswerable
queries (post-2017 or fictional films) have none.
"""

TEST_SET = [
    {
        "query": "Who directed Inception?",
        "gold_titles": ["Inception"],
        "category": "standard",
        "reference_answer": "Christopher Nolan directed Inception.",
    },
    {
        "query": "What is the plot of The Matrix?",
        "gold_titles": ["The Matrix"],
        "category": "standard",
        "reference_answer": "The Matrix is about a computer hacker who learns reality is a simulated world and joins rebels fighting its controllers.",
    },
    {
        "query": "Who directed Pulp Fiction?",
        "gold_titles": ["Pulp Fiction"],
        "category": "standard",
        "reference_answer": "Quentin Tarantino directed Pulp Fiction.",
    },
    {
        "query": "Which movie features a clownfish named Nemo?",
        "gold_titles": ["Finding Nemo"],
        "category": "standard",
        "reference_answer": "Finding Nemo features the clownfish Nemo.",
    },
    {
        "query": "Which movie follows a young lion prince reclaiming his kingdom after his father dies?",
        "gold_titles": ["The Lion King"],
        "category": "standard",
        "reference_answer": "The Lion King follows a young lion prince reclaiming his kingdom.",
    },
    {
        "query": "Which film is about dreams, corporate secrets, and a thief who enters people's minds?",
        "gold_titles": ["Inception"],
        "category": "adversarial",
        "reference_answer": "Inception is about dreams, corporate secrets, and a thief who enters minds.",
    },
    {
        "query": "What movie is about romance and disaster aboard the doomed ship Titanic?",
        "gold_titles": ["Titanic"],
        "category": "adversarial",
        "reference_answer": "Titanic is the romance and disaster movie aboard the doomed ship.",
    },
    {
        "query": "Which film has a ring that must be destroyed in Mount Doom?",
        "gold_titles": ["The Lord of the Rings: The Fellowship of the Ring"],
        "category": "adversarial",
        "reference_answer": "The Lord of the Rings: The Fellowship of the Ring involves the quest to destroy the ring.",
    },
    {
        "query": "What movie is about a shark terrorizing Amity Island?",
        "gold_titles": ["Jaws"],
        "category": "adversarial",
        "reference_answer": "Jaws is about a shark terrorizing Amity Island.",
    },
    {
        "query": "Which film features a chocolate factory run by Willy Wonka?",
        "gold_titles": ["Willy Wonka & the Chocolate Factory"],
        "category": "adversarial",
        "reference_answer": "Willy Wonka & the Chocolate Factory features Willy Wonka's chocolate factory.",
    },
    {
        "query": "Who directed Alien?",
        "gold_titles": ["Alien"],
        "category": "sequel_confusion",
        "reference_answer": "Ridley Scott directed Alien.",
    },
    {
        "query": "Who directed Toy Story?",
        "gold_titles": ["Toy Story"],
        "category": "sequel_confusion",
        "reference_answer": "John Lasseter directed Toy Story.",
    },
    {
        "query": "What is the plot of The Godfather?",
        "gold_titles": ["The Godfather"],
        "category": "sequel_confusion",
        "reference_answer": "The Godfather follows the aging patriarch of an organized crime dynasty transferring control to his reluctant son.",
    },
    {
        "query": "Who directed Terminator 2: Judgment Day?",
        "gold_titles": ["Terminator 2: Judgment Day"],
        "category": "standard",
        "reference_answer": "James Cameron directed Terminator 2: Judgment Day.",
    },
    {
        "query": "Which animated movie is about toys that come alive when humans are not around?",
        "gold_titles": ["Toy Story"],
        "category": "adversarial",
        "reference_answer": "Toy Story is about toys that come alive when humans are not around.",
    },
    {
        "query": "Who directed Oppenheimer (2023)?",
        "gold_titles": [],
        "category": "unanswerable",
        "reference_answer": "This is not answerable from the pre-July 2017 movie corpus.",
    },
    {
        "query": "What is the plot of Dune: Part Two?",
        "gold_titles": [],
        "category": "unanswerable",
        "reference_answer": "This is not answerable from the pre-July 2017 movie corpus.",
    },
    {
        "query": "Who stars in Everything Everywhere All at Once?",
        "gold_titles": [],
        "category": "unanswerable",
        "reference_answer": "This is not answerable from the pre-July 2017 movie corpus.",
    },
    {
        "query": "What is the story of the fictional film Moon Castle 2099?",
        "gold_titles": [],
        "category": "unanswerable",
        "reference_answer": "This is not answerable from the pre-July 2017 movie corpus.",
    },
    {
        "query": "Which movie follows archaeologist Indiana Jones searching for the Ark of the Covenant?",
        "gold_titles": ["Raiders of the Lost Ark"],
        "category": "adversarial",
        "reference_answer": "Raiders of the Lost Ark follows Indiana Jones searching for the Ark of the Covenant.",
    },
]


def answerable(test_set=TEST_SET):
    return [item for item in test_set if item["gold_titles"]]


def unanswerable(test_set=TEST_SET):
    return [item for item in test_set if not item["gold_titles"]]
