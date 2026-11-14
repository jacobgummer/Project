class LLMAgent:
    def __init__(self, model_name, persona_prompt=""):
        self.model_name = model_name
        self.persona_prompt = persona_prompt  # e.g., "You are a selfish strategist..."

    def get_system_prompt(self):
        base_system = (
            "Context: Player A and player B are playing a multi-round game.\n"
            "At each turn player A and player B simultaneously perform one of the following actions: {'Cooperate', 'Defect'}.\n"
            f"{self.persona_prompt}\n"
            "Remember that a player's objective is to get the highest possible number of points in the long run."
        )
        return base_system

    def choose_action(self, history_window):
        # Format prompt with history window (e.g., last 10 rounds) and query Hugging Face model
        # Return {"action": "Cooperate" or "Defect", "reason": "..."}
        pass