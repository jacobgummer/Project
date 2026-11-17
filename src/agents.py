from dataclasses import dataclass, field
import re
from typing import Literal
from ollama import chat

Move = Literal["C", "D"]

ACTION_TO_LABEL: dict[Move, str] = {
    "C": "Cooperate",
    "D": "Defect",
}


@dataclass
class RoundRecord:
    round_num: int
    llm_move: Move
    opp_move: Move
    llm_payoff: int
    opp_payoff: int


@dataclass
class AgentResponse:
    move: Move | None
    raw_content: str
    thinking: str
    attempts: int


# Standard Prisoner's Dilemma payoff matrix: (R=3, T=5, S=0, P=1)
PAYOFF_MATRIX: dict[tuple[Move, Move], tuple[int, int]] = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}


def parse_move(text: str) -> Move | None:
    """Extracts 'Cooperate' or 'Defect' (or 'C'/'D') and normalizes to 'C' or 'D'."""
    cleaned = text.strip()
    if not cleaned:
        return None

    # 1. Exact or lightly punctuated match: "Cooperate", "Defect", "\"Cooperate\"", "Action: Defect"
    strict_match = re.fullmatch(
        r"(?:action|move|choice)?\s*[:\-]?\s*[\*\"'\(\[\{]*(COOPERATE|DEFECT|C|D)[\*\"'\)\]\}\.\!\s]*",
        cleaned,
        flags=re.IGNORECASE,
    )
    if strict_match:
        token = strict_match.group(1).upper()
        return "C" if token in ("C", "COOPERATE") else "D"

    # 2. Fallback for short responses (<= 4 words) where the first or last word is the action
    words = re.findall(r"[A-Za-z]+", cleaned)
    if len(words) <= 4 and words:
        for candidate in (words[0].upper(), words[-1].upper()):
            if candidate in ("C", "COOPERATE"):
                return "C"
            if candidate in ("D", "DEFECT"):
                return "D"

    return None


@dataclass
class LLMAgent:
    model: str
    persona_name: str
    persona_prompt: str = ""
    total_rounds: int | None = 100
    think: bool = False
    temperature: float = 0.7
    max_retries: int = 2
    keep_alive: str = "10m"
    history: list[RoundRecord] = field(default_factory=list)

    def reset(self) -> None:
        """Clears round history before starting a new game."""
        self.history.clear()

    def record_round(self, llm_move: Move, opp_move: Move) -> RoundRecord:
        """Computes payoffs from the matrix and appends the round to history."""
        llm_payoff, opp_payoff = PAYOFF_MATRIX[(llm_move, opp_move)]
        record = RoundRecord(
            round_num=len(self.history) + 1,
            llm_move=llm_move,
            opp_move=opp_move,
            llm_payoff=llm_payoff,
            opp_payoff=opp_payoff,
        )
        self.history.append(record)
        return record

    def _build_system_prompt(self) -> str:
        rounds_line = (
            f"They will play a total of {self.total_rounds} rounds of this game.\n"
            if self.total_rounds is not None
            else ""
        )
        persona_line = (
            f"\nYou are playing as Player A with the following persona: {self.persona_prompt}\n"
            if self.persona_prompt.strip()
            else "\nYou are playing as Player A.\n"
        )

        return (
            "Context: Player A and player B are playing a multi-round game.\n"
            "At each turn player A and player B simultaneously perform one of the "
            'following actions: {"Cooperate", "Defect"}\n'
            "The payoffs for each combination of chosen actions are the following:\n"
            'If A plays "Defect" and B plays "Defect", A collects 1 points and B collects 1 points.\n'
            'If A plays "Defect" and B plays "Cooperate", A collects 5 points and B collects 0 points.\n'
            'If A plays "Cooperate" and B plays "Defect", A collects 0 points and B collects 5 points.\n'
            'If A plays "Cooperate" and B plays "Cooperate", A collects 3 points and B collects 3 points.\n'
            f"{rounds_line}"
            "Remember that a player's objective is to get the highest possible number of points in the long run."
            f"{persona_line}"
            'Respond with ONLY one word: "Cooperate" or "Defect". '
            "Do not include any explanation or extra text."
        )

    def _build_user_prompt(self) -> str:
        current_round = len(self.history) + 1

        if not self.history:
            return (
                f"Round {current_round}: No previous rounds have been played.\n"
                'Which action does Player A choose? ("Cooperate" or "Defect"):'
            )

        a_total = sum(r.llm_payoff for r in self.history)
        b_total = sum(r.opp_payoff for r in self.history)

        history_lines = [
            f'Round {r.round_num}: Player A played "{ACTION_TO_LABEL[r.llm_move]}" '
            f'and Player B played "{ACTION_TO_LABEL[r.opp_move]}". '
            f"Player A collected {r.llm_payoff} points and Player B collected {r.opp_payoff} points."
            for r in self.history
        ]

        return (
            "History of previous rounds:\n"
            + "\n".join(history_lines)
            + f"\nTotal points so far -> Player A: {a_total}, Player B: {b_total}.\n"
            f'Round {current_round}: Which action does Player A choose? ("Cooperate" or "Defect"):'
        )

    def choose_move(self) -> AgentResponse:
        """Queries Ollama for the next action, re-sampling cleanly if output parsing fails."""
        base_messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_prompt()},
        ]

        is_deepseek_r1 = "deepseek-r1" in self.model.lower()
        if not self.think and is_deepseek_r1:
            base_messages.append({"role": "assistant", "content": "<think>\n</think>\n\n"})

        # "Cooperate" is 1-2 tokens, so 16 tokens is plenty when think=False
        num_predict = 32768 if self.think else 16

        last_content = ""
        last_thinking = ""

        for attempt in range(1, self.max_retries + 2):
            response = chat(
                model=self.model,
                messages=base_messages,
                think=self.think,
                keep_alive=self.keep_alive,
                options={
                    "num_predict": num_predict,
                    "temperature": self.temperature,
                },
            )

            last_content = (response.message.content or "").strip()
            last_thinking = (getattr(response.message, "thinking", None) or "").strip()

            cleaned_content = re.sub(
                r"<think>.*?</think>", "", last_content, flags=re.DOTALL
            ).strip()

            move = parse_move(cleaned_content)
            if move is not None:
                return AgentResponse(
                    move=move,
                    raw_content=cleaned_content,
                    thinking=last_thinking,
                    attempts=attempt,
                )

        return AgentResponse(
            move=None,
            raw_content=last_content,
            thinking=last_thinking,
            attempts=self.max_retries + 1,
        )


if __name__ == "__main__":
    personas = {
        "baseline": "",  # Matches the paper's neutral baseline (no persona constraint)
        "cooperative": "You prioritize mutual cooperation and trust, forgiving occasional mistakes.",
    }

    for model_name in ("qwen3:1.7b", "deepseek-r1:1.5b"):
        agent = LLMAgent(
            model=model_name,
            persona_name="baseline",
            persona_prompt=personas["baseline"],
            total_rounds=100,
            think=False,
        )
        res1 = agent.choose_move()
        print(f"[{model_name}] Round 1 -> move={res1.move}, raw={res1.raw_content!r}")

        if res1.move:
            agent.record_round(llm_move=res1.move, opp_move="D")
            res2 = agent.choose_move()
            print(f"[{model_name}] Round 2 (after B=Defect) -> move={res2.move}, raw={res2.raw_content!r}")