import gymnasium
import numpy as np
from collections import deque
import cv2
from einops import rearrange
import copy


class LifeLossInfo(gymnasium.Wrapper):
    """Wrapper for Atari environments to track life loss"""

    def __init__(self, env):
        super().__init__(env)
        self.lives_info = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_lives_info = info.get("lives", 0)
        if current_lives_info < self.lives_info:
            info["life_loss"] = True
            self.lives_info = info["lives"]
        else:
            info["life_loss"] = False

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.lives_info = info.get("lives", 0)
        info["life_loss"] = False
        return observation, info


class MuJoCoRenderWrapper(gymnasium.Wrapper):
    """Wrapper to render MuJoCo state as RGB images for vision-based models"""

    def __init__(self, env, width=64, height=64, camera_id=None):
        super().__init__(env)
        self.width = width
        self.height = height
        self.camera_id = camera_id
        # Override observation space to be image-based
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )

    def _render_obs(self):
        """Render the current state as an RGB image"""
        # Use the render method with rgb_array mode
        img = self.env.render()
        if img is None:
            # Fallback if render returns None
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            # Resize to target dimensions
            img = cv2.resize(img, (self.width, self.height))
        return img

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        obs = self._render_obs()
        return obs, info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        obs = self._render_obs()
        return obs, reward, terminated, truncated, info


class RewardNormalizationWrapper(gymnasium.Wrapper):
    """Wrapper to normalize rewards using running statistics"""

    def __init__(self, env, clip_range=10.0):
        super().__init__(env)
        self.clip_range = clip_range
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0
        self.epsilon = 1e-8

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Update statistics
        self.count += 1
        delta = reward - self.running_mean
        self.running_mean += delta / self.count
        delta2 = reward - self.running_mean
        self.running_var += delta * delta2

        # Normalize reward
        if self.count > 1:
            std = np.sqrt(self.running_var / (self.count - 1) + self.epsilon)
            normalized_reward = np.clip(reward / std, -self.clip_range, self.clip_range)
        else:
            normalized_reward = reward

        return obs, normalized_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class DiscreteActionWrapper(gymnasium.Wrapper):
    """Wrapper to discretize continuous action spaces for MuJoCo environments

    Converts Box action space to Discrete by binning each dimension.
    """

    def __init__(self, env, num_bins=11):
        super().__init__(env)
        self.num_bins = num_bins

        assert isinstance(
            env.action_space, gymnasium.spaces.Box
        ), "DiscreteActionWrapper only works with Box action spaces"

        self.original_action_space = env.action_space
        self.action_dim = env.action_space.shape[0]

        # Create discrete action space: num_bins^action_dim total actions
        # For simplicity, we use a single discrete value and decode it
        self.action_space = gymnasium.spaces.Discrete(num_bins**self.action_dim)

        # Precompute action bins for efficiency
        self.action_bins = []
        for i in range(self.action_dim):
            low = self.original_action_space.low[i]
            high = self.original_action_space.high[i]
            bins = np.linspace(low, high, num_bins)
            self.action_bins.append(bins)

    def _discrete_to_continuous(self, discrete_action):
        """Convert discrete action index to continuous action vector"""
        continuous_action = np.zeros(self.action_dim, dtype=np.float32)
        action_idx = discrete_action

        for i in range(self.action_dim - 1, -1, -1):
            bin_idx = action_idx % self.num_bins
            action_idx = action_idx // self.num_bins
            continuous_action[i] = self.action_bins[i][bin_idx]

        return continuous_action

    def step(self, action):
        # Convert discrete action to continuous
        continuous_action = self._discrete_to_continuous(action)
        return self.env.step(continuous_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class SeedEnvWrapper(gymnasium.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.seed = seed
        self.env.action_space.seed(seed)

    def reset(self, **kwargs):
        kwargs["seed"] = self.seed
        obs, _ = self.env.reset(**kwargs)
        return obs, _

    def step(self, action):
        return self.env.step(action)


class MaxLast2FrameSkipWrapper(gymnasium.Wrapper):
    """Frame skipping wrapper for Atari environments"""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs, _

    def step(self, action):
        total_reward = 0
        self.obs_buffer = deque(maxlen=2)
        for _ in range(self.skip):
            obs, reward, done, truncated, info = self.env.step(action)
            self.obs_buffer.append(obs)
            total_reward += reward
            if done or truncated:
                break
        if len(self.obs_buffer) == 1:
            obs = self.obs_buffer[0]
        else:
            obs = np.max(np.stack(self.obs_buffer), axis=0)
        return obs, total_reward, done, truncated, info


def is_atari_env(env_name):
    """Check if environment is an Atari game"""
    return env_name.startswith("ALE/") or "Atari" in env_name


def is_mujoco_env(env_name):
    """Check if environment is a MuJoCo environment"""
    mujoco_envs = [
        "Hopper",
        "Walker2d",
        "HalfCheetah",
        "Ant",
        "Humanoid",
        "Swimmer",
        "Reacher",
        "Pusher",
        "InvertedPendulum",
        "InvertedDoublePendulum",
        "HumanoidStandup",
    ]
    return any(mujoco_name in env_name for mujoco_name in mujoco_envs)


def build_single_env_atari(env_name, image_size, seed):
    """Build a single Atari environment with appropriate wrappers"""
    env = gymnasium.make(
        env_name, full_action_space=False, render_mode="rgb_array", frameskip=1
    )
    env = SeedEnvWrapper(env, seed=seed)
    env = MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = LifeLossInfo(env)
    return env


def build_single_env_mujoco(
    env_name, image_size, seed, use_reward_norm=True, num_action_bins=11
):
    """Build a single MuJoCo environment with appropriate wrappers"""
    env = gymnasium.make(env_name, render_mode="rgb_array")
    env = SeedEnvWrapper(env, seed=seed)
    # Discretize continuous action space for compatibility with discrete agent
    env = DiscreteActionWrapper(env, num_bins=num_action_bins)
    # Render state as images for vision-based model
    env = MuJoCoRenderWrapper(env, width=image_size, height=image_size)
    if use_reward_norm:
        env = RewardNormalizationWrapper(env, clip_range=10.0)
    return env


def build_single_env(env_name, image_size, seed):
    """
    Automatically detect environment type and build with appropriate wrappers

    Args:
        env_name: Name of the gymnasium environment
        image_size: Target image size (assumes square images)
        seed: Random seed for reproducibility

    Returns:
        Wrapped gymnasium environment
    """
    if is_atari_env(env_name):
        print(f"🎮 Building Atari environment: {env_name}")
        return build_single_env_atari(env_name, image_size, seed)
    elif is_mujoco_env(env_name):
        print(f"🤖 Building MuJoCo environment: {env_name}")
        return build_single_env_mujoco(env_name, image_size, seed)
    else:
        # Default fallback: try to create as-is with minimal wrappers
        print(f"⚠️  Unknown environment type: {env_name}, using minimal wrappers")
        env = gymnasium.make(env_name)
        env = SeedEnvWrapper(env, seed=seed)
        return env


def build_vec_env(env_list, image_size, num_envs):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    assert num_envs % len(env_list) == 0
    env_fns = []
    vec_env_names = []
    for env_name in env_list:

        def lambda_generator(env_name, image_size):
            return lambda: build_single_env(env_name, image_size)

        env_fns += [
            lambda_generator(env_name, image_size)
            for i in range(num_envs // len(env_list))
        ]
        vec_env_names += [env_name for i in range(num_envs // len(env_list))]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env, vec_env_names


if __name__ == "__main__":
    vec_env, vec_env_names = build_vec_env(
        ["ALE/Pong-v5", "ALE/IceHockey-v5", "ALE/Breakout-v5", "ALE/Tennis-v5"],
        64,
        num_envs=8,
    )
    current_obs, _ = vec_env.reset()
    while True:
        action = vec_env.action_space.sample()
        obs, reward, done, truncated, info = vec_env.step(action)
        # done = done or truncated
        if done.any():
            print("---------")
            print(reward)
            print(info["episode_frame_number"])
        cv2.imshow("Pong", current_obs[0])
        cv2.imshow("IceHockey", current_obs[2])
        cv2.imshow("Breakout", current_obs[4])
        cv2.imshow("Tennis", current_obs[6])
        cv2.waitKey(40)
        current_obs = obs
