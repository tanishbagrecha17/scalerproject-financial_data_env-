from env import FinancialDataEnv
from models import Action

env = FinancialDataEnv("datasets/sample.csv")

obs = env.reset()
print("Initial Observation:\n", obs)

# Step 1
action = Action(action_type="remove_duplicates")
obs, reward, done, _ = env.step(action)

print("\nAfter Action 1:")
print(obs)
print("Reward:", reward)

# Step 2
action = Action(action_type="fix_negative_amount")
obs, reward, done, _ = env.step(action)

print("\nAfter Action 2:")
print(obs)
print("Reward:", reward)
from env import FinancialDataEnv
from models import Action
from tasks import EASY_TASK

env = FinancialDataEnv("datasets/sample.csv", task=EASY_TASK)

obs = env.reset()
print("Task:", env.task.description)
print("Initial:", obs)

# Step 1
action = Action(action_type="remove_duplicates")
obs, reward, done, _ = env.step(action)

print("\nAfter Action 1:")
print(obs)
print("Reward:", reward)
obs, reward, done, info = env.step(action)

print("Reward:", reward)
print("Final Score:", info["final_score"])