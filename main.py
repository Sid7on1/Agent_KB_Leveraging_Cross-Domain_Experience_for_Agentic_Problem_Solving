import numpy as np
import random

class AgentKB:
    """
    Implements a simplified Agent Knowledge Base (AgentKB) for knowledge sharing
    and transfer between agents.  Focuses on the Reason-Retrieve-Refine pipeline.

    This simplified version uses a dictionary to store experiences and retrieves
    relevant experiences based on a simple similarity metric.  Refinement is
    simulated by adjusting the agent's action based on retrieved experiences.
    """

    def __init__(self):
        """
        Initializes the AgentKB with an empty knowledge base.
        """
        self.knowledge_base = {}  # Stores experiences: {task_description: [(state, action, reward)]}

    def store_experience(self, task_description, state, action, reward):
        """
        Stores an agent's experience in the knowledge base.

        Args:
            task_description (str): A description of the task the agent was performing.
            state (np.ndarray): The agent's state.
            action (int): The action the agent took.
            reward (float): The reward the agent received.
        """
        if task_description not in self.knowledge_base:
            self.knowledge_base[task_description] = []
        self.knowledge_base[task_description].append((state, action, reward))

    def retrieve_relevant_experiences(self, task_description, current_state, top_k=3):
        """
        Retrieves the most relevant experiences from the knowledge base based on
        task description and state similarity.  Uses a simple Euclidean distance
        for state similarity.

        Args:
            task_description (str): The description of the current task.
            current_state (np.ndarray): The agent's current state.
            top_k (int): The number of top experiences to retrieve.

        Returns:
            list: A list of the top_k most relevant experiences (state, action, reward) tuples.
                   Returns an empty list if no relevant experiences are found.
        """
        if task_description not in self.knowledge_base:
            return []

        experiences = self.knowledge_base[task_description]
        if not experiences:
            return []

        # Calculate state similarity (Euclidean distance)
        similarities = [np.linalg.norm(current_state - state) for state, _, _ in experiences]

        # Sort experiences by similarity (lower distance = more similar)
        sorted_experiences = sorted(zip(similarities, experiences), key=lambda x: x[0])

        # Return the top_k most relevant experiences
        top_experiences = [exp for _, exp in sorted_experiences[:top_k]]
        return top_experiences

    def refine_action(self, task_description, current_state, proposed_action):
        """
        Refines the proposed action based on retrieved experiences.  This is a
        simplified refinement process that adjusts the action based on the average
        reward of similar actions in retrieved experiences.

        Args:
            task_description (str): The description of the current task.
            current_state (np.ndarray): The agent's current state.
            proposed_action (int): The action the agent is considering taking.

        Returns:
            int: The refined action.
        """
        relevant_experiences = self.retrieve_relevant_experiences(task_description, current_state)

        if not relevant_experiences:
            return proposed_action  # No relevant experiences, return original action

        # Calculate average reward for the proposed action in relevant experiences
        rewards_for_action = [reward for state, action, reward in relevant_experiences if action == proposed_action]
        if rewards_for_action:
            avg_reward = np.mean(rewards_for_action)
        else:
            avg_reward = -1  # Penalize if the action wasn't taken in relevant experiences

        # Suggest a different action if the average reward for the proposed action is low
        if avg_reward < 0:
            # Find the action with the highest average reward in relevant experiences
            action_rewards = {}
            for state, action, reward in relevant_experiences:
                if action not in action_rewards:
                    action_rewards[action] = []
                action_rewards[action].append(reward)

            best_action = proposed_action  # Default to the proposed action
            best_avg_reward = avg_reward

            for action, rewards in action_rewards.items():
                current_avg_reward = np.mean(rewards)
                if current_avg_reward > best_avg_reward:
                    best_action = action
                    best_avg_reward = current_avg_reward

            return best_action
        else:
            return proposed_action  # Keep the proposed action if it has a good average reward


if __name__ == '__main__':
    # Example Usage
    agent_kb = AgentKB()

    # Agent 1 experiences
    agent_kb.store_experience("navigation", np.array([1, 2]), 0, 0.5)  # State, action, reward
    agent_kb.store_experience("navigation", np.array([2, 3]), 1, 0.8)
    agent_kb.store_experience("navigation", np.array([1, 1]), 0, 0.6)

    # Agent 2 experiences (different task)
    agent_kb.store_experience("object_pickup", np.array([0.5, 0.5]), 2, 0.9)
    agent_kb.store_experience("object_pickup", np.array([0.6, 0.4]), 3, 0.7)

    # New agent facing a navigation task
    current_state = np.array([1.5, 2.5])
    proposed_action = 0

    # Refine the action using the AgentKB
    refined_action = agent_kb.refine_action("navigation", current_state, proposed_action)

    print(f"Proposed action: {proposed_action}")
    print(f"Refined action: {refined_action}")

    # Example of no relevant experiences
    refined_action_no_exp = agent_kb.refine_action("unknown_task", current_state, proposed_action)
    print(f"Refined action (no relevant experiences): {refined_action_no_exp}")