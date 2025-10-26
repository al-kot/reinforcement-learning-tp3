"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import time
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np

from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SARSAAgent

env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore


def record_video(agent, video_name: str):
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo

    video_env = RecordVideo(
        gym.make("Taxi-v3", render_mode="rgb_array"),
        video_folder="./videos",
        name_prefix=video_name,
    )

    old_eps = agent.epsilon
    agent.epsilon = 0.0

    state, _ = video_env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_best_action(state)
        state, reward, done, _, _ = video_env.step(action)
        total_reward += float(reward)

    video_env.close()
    print(f"Video saved for {video_name} — reward = {total_reward}")

    agent.epsilon = old_eps


#################################################
# 1. Play with QLearningAgent
#################################################

# You can edit these hyperparameters!
agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for _ in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        # BEGIN SOLUTION
        agent.update(s, a, r, next_s)

        total_reward += float(r)

        s = next_s

        if done:
            break
        # END SOLUTION

    return total_reward


rewards = []
start_time = time.time()
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
print(f"qlearning training time: {time.time() - start_time}")
rewards_qlearning = rewards

assert np.mean(rewards[-100:]) > 0.0
# TODO: créer des vidéos de l'agent en action
record_video(agent, "qlearning")


#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################


agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

rewards = []
start_time = time.time()
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
print(f"qlearning_eps training time: {time.time() - start_time}")
rewards_qlearning_eps = rewards

assert np.mean(rewards[-100:]) > 0.0

# TODO: créer des vidéos de l'agent en action
record_video(agent, "qlearning_eps_scheduling")


####################
# 3. Play with SARSA
####################


agent = SARSAAgent(
    learning_rate=0.5, gamma=0.99, epsilon=0.2, legal_actions=list(range(n_actions))
)

rewards = []
start_time = time.time()
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward (sarsa)", np.mean(rewards[-100:]))
print(f"sarsa training time: {time.time() - start_time}")
rewards_sarsa = rewards

record_video(agent, "sarsa")

plt.figure(figsize=(10, 6))
plt.plot(rewards_sarsa, label="SARSA")
plt.plot(rewards_qlearning, label="Q-Learning")
plt.plot(rewards_qlearning_eps, label="Q-Learning Eps Scheduling")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Comparaison des performances sur Taxi-v3")
plt.legend()
plt.grid(True)
plt.savefig("learning_curves.png", dpi=300)
