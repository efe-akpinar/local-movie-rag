"""Simulated users asking a mix of example questions, with think time between requests."""

import random

from locust import HttpUser, between, task

QUESTIONS = [
    "Who directed Toy Story?",
    "What movie is about a shark terrorizing Amity Island?",
    "Which movie features a clownfish named Nemo?",
    "What is the plot of The Godfather?",
    "Who directed Oppenheimer (2023)?",
]


class MovieQuestionUser(HttpUser):
    wait_time = between(2, 6)

    @task(10)
    def ask(self):
        question = random.choice(QUESTIONS)
        with self.client.post("/ask", json={"question": question}, name="/ask", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            elif not response.json().get("answer"):
                response.failure("empty answer")
            else:
                response.success()

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
