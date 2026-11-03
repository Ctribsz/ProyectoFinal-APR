from gymnasium.envs.registration import register

from envs.balance_bot_env import BalanceBotEnv, load_config

# Sin max_episode_steps: el entorno trunca por su cuenta según el YAML del nivel.
register(id="BalanceBot-v0", entry_point="envs.balance_bot_env:BalanceBotEnv")

__all__ = ["BalanceBotEnv", "load_config"]
