import torch
import cv2
import numpy as np
import time
import os
import re
from collections import deque
from env import PcGameEnv
from train import DQN, FRAMES_STACK

MODEL_PATH = os.path.join("checkpoints", "dqn_checkpoint")
TEST_TARGET_STEP_TIME_SEC = 0.12
TEST_PACING_SPIN_THRESHOLD_SEC = 0.002
TEST_EPISODES = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_path(default_path: str) -> str:
    if os.path.isfile(default_path):
        return default_path

    if os.path.isdir(default_path):
        ckpt_dir = default_path
    else:
        ckpt_dir = os.path.dirname(default_path) or "."

    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    checkpoint_pattern = re.compile(r"^dqn_checkpoint(?:\.pth)?_(\d+)\.pth$")

    def checkpoint_sort_key(path: str) -> tuple:
        match = checkpoint_pattern.fullmatch(os.path.basename(path))
        step = int(match.group(1)) if match else -1
        return step, os.path.getmtime(path)

    candidates = [os.path.join(ckpt_dir, name) for name in os.listdir(ckpt_dir) if checkpoint_pattern.fullmatch(name)]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint file found in {ckpt_dir}. Expected {default_path} or dqn_checkpoint_*.pth / dqn_checkpoint.pth_*.pth"
        )

    return max(candidates, key=checkpoint_sort_key)


def wait_until_precise(deadline: float, spin_threshold_sec: float) -> None:
    """Wait until deadline with better precision than plain time.sleep on Windows."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            return
        if remaining > spin_threshold_sec:
            time.sleep(remaining - spin_threshold_sec)
            continue
        break

    while time.perf_counter() < deadline:
        pass


def print_score_stats(episode_scores: list[float]) -> None:
    if not episode_scores:
        return
    mean_score = float(np.mean(episode_scores))
    std_score = float(np.std(episode_scores))
    print("Score stats: " f"episodes={len(episode_scores)}, avg={mean_score:.3f}, std={std_score:.3f}")


def main():
    env = PcGameEnv(target_train_step_time_sec=TEST_TARGET_STEP_TIME_SEC)

    obs, _ = env.reset()
    if obs.ndim == 2:
        per_frame_channels = 1
        frame_h, frame_w = obs.shape
    elif obs.ndim == 3:
        per_frame_channels, frame_h, frame_w = obs.shape
    else:
        raise RuntimeError(f"Unsupported observation shape from env: {obs.shape}")

    model_input_shape = (per_frame_channels * FRAMES_STACK, frame_h, frame_w)

    # Initialize Model
    model = DQN(model_input_shape, env.action_space.n).to(device)

    try:
        model_path = resolve_model_path(MODEL_PATH)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {MODEL_PATH} not found. Please train first.")
        return
    except RuntimeError as e:
        print(
            "Checkpoint shape mismatch. This usually means the checkpoint was "
            "trained with old observation channels. Please retrain with current env."
        )
        print(f"Detail: {e}")
        return

    model.eval()

    frames = deque([obs] * FRAMES_STACK, maxlen=FRAMES_STACK)
    total_reward = 0.0
    episode_step_count = 0
    episode_step_time_sum = 0.0
    episode_score = 0.0
    episode_scores = []
    completed_episodes = 0

    if env.target_train_step_time_sec is None:
        print("Test step pacing: disabled")
    else:
        print(
            "Test step pacing target: "
            f"{env.target_train_step_time_sec:.3f}s (spin_tail={TEST_PACING_SPIN_THRESHOLD_SEC:.3f}s)"
        )

    print(f"Starting testing... episodes={TEST_EPISODES}. " "Press Ctrl+C to stop early.")
    try:
        while completed_episodes < TEST_EPISODES:
            step_start = time.perf_counter()

            # Epsilon = 0 (Greedy)
            with torch.no_grad():
                if obs.ndim == 2:
                    stacked_obs = np.stack(frames, axis=0)
                else:
                    stacked_obs = np.concatenate(list(frames), axis=0)
                state_tensor = torch.from_numpy(stacked_obs).unsqueeze(0).float().to(device) / 255.0
                q_value = model(state_tensor)
                action = q_value.max(1)[1].item()
            # print(f"Q values: {q_value}")
            # print(f"action: {action}")
            obs, reward, done, truncated, info = env.step(action)
            frames.append(obs)
            total_reward += reward
            episode_score += info.get("score_reward", 0.0)
            # # Optional: Show what the agent sees
            # if obs.ndim == 2:
            #     preview = obs
            # else:
            #     gray_preview = obs[0]
            #     mask_preview = (
            #         obs[1] if obs.shape[0] > 1 else np.zeros_like(gray_preview)
            #     )
            #     preview = np.concatenate([gray_preview, mask_preview], axis=1)

            # cv2.imshow("Agent View", preview)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

            step_time_raw = time.perf_counter() - step_start
            target_step_time = env.target_train_step_time_sec
            if target_step_time is not None and step_time_raw < target_step_time:
                wait_until_precise(
                    step_start + target_step_time,
                    TEST_PACING_SPIN_THRESHOLD_SEC,
                )

            step_time = time.perf_counter() - step_start
            episode_step_count += 1
            episode_step_time_sum += step_time

            if done or truncated:
                avg_step_time = episode_step_time_sum / episode_step_count if episode_step_count > 0 else 0.0
                episode_scores.append(episode_score)
                completed_episodes += 1
                print(
                    f"Episode {completed_episodes}/{TEST_EPISODES} finished. "
                    f"Game Over. Reward: {total_reward:.3f}, "
                    f"Score: {episode_score:.3f}, "
                    f"AvgStepTime: {avg_step_time:.4f}s"
                )
                obs, _ = env.reset()
                total_reward = 0.0
                episode_step_count = 0
                episode_step_time_sum = 0.0
                episode_score = 0.0
                frames = deque([obs] * FRAMES_STACK, maxlen=FRAMES_STACK)

        print_score_stats(episode_scores)

    except KeyboardInterrupt:
        print_score_stats(episode_scores)
        print("Testing stopped.")
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
