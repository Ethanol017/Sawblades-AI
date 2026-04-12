import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import os
import time
from env import PcGameEnv
from utils.replay_buffer import ReplayBuffer


# --- Hyperparameters ---
MAX_STEPS = 100000  # 100k baseline
BATCH_SIZE = 64
LR = 5e-5
GAMMA = 0.99
EPSILON_START_STEPS = 15000  # 15k
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 60000  # 60k
TRAIN_FREQ = 4
MEMORY_CAPACITY = 50000  # 50k
LEARNING_STARTS = 15000  # 15k
MAX_GRAD_NORM = 1.0
USE_SOFT_UPDATE = True
TAU = 0.002
TARGET_UPDATE = 10000  # for hard update
FRAMES_STACK = 4
# for saving/loading model
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR + "/dqn_checkpoint.pth"
RESUME_TRAINING = False
RESUME_LOGDIR = "runs/dqn_experiment_20260409-200509"
BUFFER_PATH = "replay_buffer.pkl"
SAVE_INTERVAL = 20000
# environment window behavior
ENV_AUTO_RESIZE_WINDOW = True
ENV_AUTO_ACTIVATE_WINDOW = False
# for logging
LOG_INTERVAL = 100
HIST_INTERVAL = 1000
GRAD_NORM_TYPE = 2.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        # CNN Layers
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate FC input size
        self.fc_input_dim = self.feature_size()

        # Dueling Network Architecture
        # Value Stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Advantage Stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def feature_size(self):
        return (
            self.features(torch.autograd.Variable(torch.zeros(1, *self.input_shape)))
            .view(1, -1)
            .size(1)
        )

    def forward(self, x):
        x = self.features(x)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine V and A to get Q
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals


def compute_grad_norm(parameters, norm_type: float = 2.0) -> float:
    total_norm = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type) if total_norm > 0 else 0.0


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def compute_linear_epsilon(step: int) -> float:
    decay_end_ratio = EPSILON_DECAY_END_PERCENT / 100.0
    decay_end_step = max(1, int(MAX_STEPS * decay_end_ratio))
    progress = min(max(step, 0), decay_end_step) / decay_end_step
    return EPSILON_START + (EPSILON_END - EPSILON_START) * progress


def save_replay_buffer(buffer: ReplayBuffer, buffer_path: str) -> None:
    tmp_path = buffer_path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, buffer_path)
        print(f"Replay buffer saved to {buffer_path} (size={len(buffer)})")
    except Exception as e:
        print(f"Failed to save replay buffer to {buffer_path}: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def load_replay_buffer(
    buffer_path: str,
    expected_frame_shape: tuple,
    expected_frame_history_len: int,
) -> ReplayBuffer:
    with open(buffer_path, "rb") as f:
        loaded_buffer = pickle.load(f)

    if not isinstance(loaded_buffer, ReplayBuffer):
        raise RuntimeError(
            f"Invalid buffer object type: {type(loaded_buffer)} (expected ReplayBuffer)"
        )

    if loaded_buffer.frame_history_len != expected_frame_history_len:
        raise RuntimeError(
            "Replay buffer frame_history_len mismatch "
            f"(buffer={loaded_buffer.frame_history_len}, expected={expected_frame_history_len})"
        )

    if loaded_buffer.obs is not None:
        loaded_frame_shape = tuple(loaded_buffer.obs.shape[1:])
        if loaded_frame_shape != expected_frame_shape:
            raise RuntimeError(
                "Replay buffer frame shape mismatch "
                f"(buffer={loaded_frame_shape}, expected={expected_frame_shape})"
            )

    return loaded_buffer


def main():
    try:
        env = PcGameEnv(
            auto_resize_window=ENV_AUTO_RESIZE_WINDOW,
            auto_activate_window=ENV_AUTO_ACTIVATE_WINDOW,
        )
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        return

    obs_shape = env.observation_space.shape
    if len(obs_shape) == 2:
        per_frame_channels = 1
        frame_h, frame_w = obs_shape
    elif len(obs_shape) == 3:
        per_frame_channels, frame_h, frame_w = obs_shape
    else:
        raise RuntimeError(f"Unsupported observation shape: {obs_shape}")

    model_input_shape = (
        int(per_frame_channels) * FRAMES_STACK,
        int(frame_h),
        int(frame_w),
    )
    print(
        f"Observation shape from env: {obs_shape}, model input shape: {model_input_shape}"
    )

    # Initialize Networks
    policy_net = DQN(model_input_shape, env.action_space.n).to(device)
    target_net = DQN(model_input_shape, env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    policy_net.train()
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(MEMORY_CAPACITY, FRAMES_STACK)
    expected_buffer_frame_shape = (int(per_frame_channels), int(frame_h), int(frame_w))

    # TensorBoard Writer
    if RESUME_TRAINING:
        log_dir = RESUME_LOGDIR
    else:
        log_dir = f"runs/dqn_experiment_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    # Create checkpoint directory
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    epsilon = EPSILON_START
    steps_done = 0

    # Load checkpoint if exists
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
        checkpoint_input_shape = tuple(checkpoint.get("model_input_shape", ()))
        if checkpoint_input_shape and checkpoint_input_shape != model_input_shape:
            print(
                "Checkpoint model_input_shape mismatch "
                f"(checkpoint={checkpoint_input_shape}, current={model_input_shape}); "
                "skip loading and train from scratch."
            )
        else:
            try:
                policy_net.load_state_dict(checkpoint["model_state_dict"])
                target_net.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                steps_done = checkpoint.get("steps_done", 0)
                epsilon = compute_linear_epsilon(steps_done)
            except RuntimeError as e:
                print(
                    "Checkpoint load failed, likely due to input channel mismatch. "
                    f"Train from scratch. Error: {e}"
                )

    if RESUME_TRAINING:
        if os.path.exists(BUFFER_PATH):
            print(f"Loading replay buffer from {BUFFER_PATH}")
            try:
                buffer = load_replay_buffer(
                    BUFFER_PATH,
                    expected_frame_shape=expected_buffer_frame_shape,
                    expected_frame_history_len=FRAMES_STACK,
                )
                print(f"Replay buffer restored (size={len(buffer)})")
            except Exception as e:
                print(
                    "Replay buffer load failed, start with empty buffer. " f"Error: {e}"
                )
                buffer = ReplayBuffer(MEMORY_CAPACITY, FRAMES_STACK)
        else:
            print(
                f"RESUME_TRAINING=True but {BUFFER_PATH} not found, start with empty buffer."
            )

    print("Starting training...")
    obs, _ = env.reset()
    episode_reward = 0
    episode_len = 0
    need_save = False
    last_idx = 0
    step = steps_done

    try:
        for step in range(steps_done, MAX_STEPS):
            t1 = time.time()

            # store_frame expects (H,W,C); convert from env output format.
            if obs.ndim == 2:
                obs_for_buffer = obs.reshape(frame_h, frame_w, 1)
            elif obs.ndim == 3:
                obs_for_buffer = np.transpose(obs, (1, 2, 0))
            else:
                raise RuntimeError(f"Unsupported obs ndim: {obs.ndim}")

            last_idx = buffer.store_frame(obs_for_buffer)
            recent_observations = (
                buffer.encode_recent_observation()
            )  # frame stacking (C*FRAMES_STACK, H, W)

            # 1. Epsilon-Greedy Action Selection
            if random.random() > epsilon:
                with torch.no_grad():
                    state_tensor = (
                        torch.from_numpy(recent_observations)
                        .unsqueeze(0)
                        .float()
                        .to(device)
                        / 255.0
                    )
                    q_value = policy_net(state_tensor)
                    action = q_value.max(1)[1].item()

                    # Log Average Q-value (optional, for monitoring)
                    if step % LOG_INTERVAL == 0:
                        writer.add_scalar("Q_Value/max_q", q_value.max().item(), step)
                        writer.add_scalar("Q_Value/mean_q", q_value.mean().item(), step)
                        writer.add_scalar("Q_Value/min_q", q_value.min().item(), step)
                        if step % HIST_INTERVAL == 0:
                            writer.add_histogram(
                                "Q_Value/q_values", q_value.detach().cpu().numpy(), step
                            )
            else:
                action = env.action_space.sample()

            if step % HIST_INTERVAL == 0:
                writer.add_scalar("Replay/size", len(buffer), step)

            # 2. Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_len += 1

            # 3. Push to Buffer
            buffer.store_effect(last_idx, action, reward, terminated or truncated)

            obs = next_obs

            # 4. Train
            can_train = (
                buffer.can_sample(BATCH_SIZE)
                and (step >= LEARNING_STARTS)
                and (step % TRAIN_FREQ == 0)
            )
            if can_train:
                obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = (
                    buffer.sample(BATCH_SIZE)
                )
                
                obs_batch = torch.from_numpy(obs_batch).float().to(device) / 255.0
                act_batch = torch.from_numpy(act_batch).long().to(device)
                rew_batch = torch.from_numpy(rew_batch).to(device)
                next_obs_batch = (
                    torch.from_numpy(next_obs_batch).float().to(device) / 255.0
                )
                done_batch = torch.from_numpy(done_batch).to(device)

                q_values = (
                    policy_net(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze(1)
                )

                with torch.no_grad():
                    next_actions = policy_net(next_obs_batch).argmax(1)
                    next_q_values = (
                        target_net(next_obs_batch)
                        .gather(1, next_actions.unsqueeze(1))
                        .squeeze(1)
                    )
                    expected_q_values = rew_batch + GAMMA * next_q_values * (
                        1 - done_batch
                    )

                td_error = q_values - expected_q_values  # TD Error for logging
                loss = F.smooth_l1_loss(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()

                clipped_norm = torch.nn.utils.clip_grad_norm_(
                    policy_net.parameters(), max_norm=MAX_GRAD_NORM
                )
                optimizer.step()
                if USE_SOFT_UPDATE:
                    soft_update(target_net, policy_net, TAU)

                # Log Loss
                if step % 10 == 0:
                    writer.add_scalar("Loss/train", loss.item(), step)

                if step % LOG_INTERVAL == 0:
                    writer.add_scalar(
                        "TD_Error/abs_mean", td_error.abs().mean().item(), step
                    )
                    writer.add_scalar("TD_Error/mean", td_error.mean().item(), step)
                    writer.add_scalar("Grad/clipped_norm", float(clipped_norm), step)
                    writer.add_scalar(
                        "Grad/norm",
                        compute_grad_norm(policy_net.parameters(), GRAD_NORM_TYPE),
                        step,
                    )

                if step % HIST_INTERVAL == 0:
                    writer.add_histogram(
                        "TD_Error/td_error", td_error.detach().cpu().numpy(), step
                    )
                    writer.add_histogram(
                        "Action/batch_actions", act_batch.detach().cpu().numpy(), step
                    )

            # Update Epsilon (Linear Decay)
            epsilon = compute_linear_epsilon(step)

            if step % LOG_INTERVAL == 0:
                writer.add_scalar("Epsilon", epsilon, step)

            # 5. Update Target Net
            if (not USE_SOFT_UPDATE) and (step % TARGET_UPDATE == 0):
                target_net.load_state_dict(policy_net.state_dict())

            t2 = time.time()
            writer.add_scalar("Time/step_time", t2 - t1, step)

            if step > 0 and step % SAVE_INTERVAL == 0:
                save_replay_buffer(buffer, BUFFER_PATH)

            # 6. Reset and save if done
            if step % SAVE_INTERVAL == 0:
                need_save = True
            if terminated or truncated:
                print(
                    f"Step: {step}, Episode Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}"
                )
                writer.add_scalar("Reward/episode", episode_reward, step)
                writer.add_scalar("Episode/len", episode_len, step)
                if episode_len > 0:
                    writer.add_scalar(
                        "Reward/per_step", episode_reward / episode_len, step
                    )
                # writer.add_scalar('Score/episode', info.get('score', 0), step)

                if need_save:
                    need_save = False
                    torch.save(
                        {
                            "model_state_dict": policy_net.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "steps_done": step,
                            "epsilon": epsilon,
                            "model_input_shape": model_input_shape,
                            "frame_stack": FRAMES_STACK,
                            "per_frame_channels": int(per_frame_channels),
                        },
                        CHECKPOINT_PATH + f"_{step}.pth",
                    )
                    save_replay_buffer(buffer, BUFFER_PATH)

                obs, _ = env.reset()
                episode_reward = 0
                episode_len = 0

    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint and replay buffer...")
        torch.save(
            {
                "model_state_dict": policy_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "steps_done": step,
                "epsilon": epsilon,
                "model_input_shape": model_input_shape,
                "frame_stack": FRAMES_STACK,
                "per_frame_channels": int(per_frame_channels),
            },
            CHECKPOINT_PATH,
        )
        save_replay_buffer(buffer, BUFFER_PATH)
        writer.close()
        env.close()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Saving checkpoint and replay buffer before exit...")
        torch.save(
            {
                "model_state_dict": policy_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "steps_done": step,
                "epsilon": epsilon,
                "model_input_shape": model_input_shape,
                "frame_stack": FRAMES_STACK,
                "per_frame_channels": int(per_frame_channels),
            },
            CHECKPOINT_PATH,
        )
        save_replay_buffer(buffer, BUFFER_PATH)
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
