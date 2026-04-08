from env import FinancialDataEnv
from models import Action
from tasks import EASY_TASK, MEDIUM_TASK, HARD_TASK


def run_task(task):
    print(f"\nRunning Task: {task.name}")

    env = FinancialDataEnv("datasets/sample.csv", task=task)
    obs = env.reset()

    done = False
    total_reward = 0

    actions = [
        "remove_duplicates",
        "fix_negative_amount",
        "drop_missing_customer"
    ]

    step = 0
    while not done:
        action_type = actions[step % len(actions)]
        action = Action(action_type=action_type)

        obs, reward, done, info = env.step(action)

        total_reward += reward.score
        step += 1

    print("Final Score:", info["final_score"])
    print("Total Reward:", round(total_reward, 3))


if __name__ == "__main__":
    run_task(EASY_TASK)
    run_task(MEDIUM_TASK)
    run_task(HARD_TASK)