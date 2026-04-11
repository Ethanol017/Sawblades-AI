import torch
import cv2
import numpy as np
import time
import os
from collections import deque
from env import PcGameEnv
from train import DQN, FRAMES_STACK

MODEL_PATH = "checkpoints/dqn_checkpoint.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_path(default_path: str) -> str:
    if os.path.exists(default_path):
        return default_path

    ckpt_dir = os.path.dirname(default_path) or "."
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    candidates = [
        os.path.join(ckpt_dir, name)
        for name in os.listdir(ckpt_dir)
        if name.startswith("dqn_checkpoint.pth_") and name.endswith(".pth")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint file found in {ckpt_dir}. Expected {default_path} or dqn_checkpoint.pth_*.pth"
        )

    return max(candidates, key=os.path.getmtime)


def main():
    env = PcGameEnv()

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

    print("Starting testing... Press Ctrl+C to stop.")
    try:
        while True:
            # Epsilon = 0 (Greedy)
            with torch.no_grad():
                if obs.ndim == 2:
                    stacked_obs = np.stack(frames, axis=0)
                else:
                    stacked_obs = np.concatenate(list(frames), axis=0)
                state_tensor = (
                    torch.from_numpy(stacked_obs).unsqueeze(0).float().to(device)
                    / 255.0
                )
                q_value = model(state_tensor)
                action = q_value.max(1)[1].item()
            print(f"Q values: {q_value}")
            print(f"action: {action}")
            obs, reward, done, truncated, info = env.step(action)
            frames.append(obs)

            # Optional: Show what the agent sees
            if obs.ndim == 2:
                preview = obs
            else:
                gray_preview = obs[0]
                mask_preview = (
                    obs[1] if obs.shape[0] > 1 else np.zeros_like(gray_preview)
                )
                preview = np.concatenate([gray_preview, mask_preview], axis=1)

            cv2.imshow("Agent View", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if done or truncated:
                print(f"Game Over. Reward: {reward}")
                obs, _ = env.reset()
                frames = deque([obs] * FRAMES_STACK, maxlen=FRAMES_STACK)

    except KeyboardInterrupt:
        print("Testing stopped.")
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
