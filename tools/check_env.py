import sys
import os
import cv2
import numpy as np
import mss
import time
import pygetwindow as gw

# Add parent directory to path to import env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from env import GAME_ROI, SCORE_ROI, START_CHECK_ROI, WHITE_THRESHOLD, RED_THRESHOLD, WINDOW_SETTINGS, resize_game_window
except ImportError:
    print("Error: Could not import settings from env.py. Make sure you are running this from the tools directory or the project root.")
    sys.exit(1)


def main():
    # Resize Window
    resize_game_window()

    with mss.mss() as sct:
        print("Starting Env Checker...")
        print("Press 'q' to quit.")

        while True:
            # 1. Grab the main GAME_ROI
            try:
                game_screenshot = sct.grab(GAME_ROI)
            except Exception as e:
                print(f"Error grabbing screen: {e}")
                time.sleep(1)
                continue

            img = np.array(game_screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Helper to draw ROI relative to GAME_ROI
            def draw_roi(image, roi, color, label):
                # Calculate relative coordinates
                # If ROI is outside GAME_ROI, this might draw outside the image bounds, which OpenCV handles (clips)
                x = roi['left'] - GAME_ROI['left']
                y = roi['top'] - GAME_ROI['top']
                w = roi['width']
                h = roi['height']

                # Draw rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                # Draw label background for readability
                label_size, _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (x, y - 20),
                              (x + label_size[0], y), color, -1)
                cv2.putText(image, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                return image

            # Draw SCORE_ROI (Green)
            SCORE_ROI = {'top': 173, 'left': 754, 'width': 454, 'height': 6}
            img = draw_roi(img, SCORE_ROI, (0, 255, 0), "Score")

            # Draw START_CHECK_ROI (Blue)
            img = draw_roi(img, START_CHECK_ROI, (255, 0, 0), "Start Check")

            # --- Check Logic ---

            # Preview AI Input (84x84)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            # Resize back to larger size for visualization (e.g., 256x256)
            preview_ai = cv2.resize(
                resized, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("AI Input Preview (84x84)", preview_ai)
            # Move window to avoid overlap
            cv2.moveWindow("AI Input Preview (84x84)", 850, 10)

            # --- Score Change Detection Preview ---
            # 1. Grab Score ROI
            score_img = np.array(sct.grab(SCORE_ROI))

            # 2. Process (Same as env.py - HSV Filter)
            score_bgr = cv2.cvtColor(score_img, cv2.COLOR_BGRA2BGR)
            score_hsv = cv2.cvtColor(score_bgr, cv2.COLOR_BGR2HSV)

            # 分數紅色: H=9, S=76%, V=78% (閃爍時會變暗)
            # 鋸齒棕色: H=32 (用 H<15 排除)
            # 鋸齒黑邊: S=0%, V=14% (用 S>80, V>80 排除)
            lower = np.array([0, 80, 80], dtype=np.uint8)
            upper = np.array([15, 255, 255], dtype=np.uint8)

            score_mask = cv2.inRange(score_hsv, lower, upper)

            # DEBUG: 計算白色像素數量
            white_pixels = cv2.countNonZero(score_mask)
            # print(f"White Pixels: {white_pixels}")

            # DEBUG: 顯示原始 mask (放大顯示)
            mask_preview = cv2.resize(
                score_mask, (400, 160), interpolation=cv2.INTER_NEAREST)

            score_small = cv2.resize(
                score_mask, (50, 20), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Score Mask (score_small)", score_small)

            # 3. Calculate Diff
            if 'prev_score_thresh' in locals():
                score_diff = cv2.absdiff(prev_score_thresh, score_small)
                diff_count = np.count_nonzero(score_diff)

                # Visualize Diff (Make it bigger!)
                diff_vis = cv2.resize(
                    score_diff, (400, 160), interpolation=cv2.INTER_NEAREST)
                # Add background for text
                cv2.rectangle(diff_vis, (0, 0), (400, 60), (0, 0, 0), -1)

                cv2.putText(
                    diff_vis, f"Diff: {diff_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
                if diff_count > 30 and white_pixels > 260:
                    print("GET SCORE CHANGE!")
                    cv2.putText(diff_vis, "CHANGED!", (200, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)

                cv2.imshow("Score Diff", diff_vis)
                # Move window to avoid overlap
                cv2.moveWindow("Score Diff", 850, 300)

            prev_score_thresh = score_small
            # --------------------------------------            # Check Start Status
            # We grab START_CHECK_ROI separately to be precise, or extract from game_screenshot
            # Extracting is faster and ensures sync
            start_roi_rel_x = START_CHECK_ROI['left'] - GAME_ROI['left']
            start_roi_rel_y = START_CHECK_ROI['top'] - GAME_ROI['top']

            # Ensure coordinates are valid for slicing
            if (0 <= start_roi_rel_x < GAME_ROI['width'] and
                    0 <= start_roi_rel_y < GAME_ROI['height']):

                start_slice = img[start_roi_rel_y: start_roi_rel_y + START_CHECK_ROI['height'],
                                  start_roi_rel_x: start_roi_rel_x + START_CHECK_ROI['width']]

                if start_slice.size > 0:
                    start_gray = cv2.cvtColor(start_slice, cv2.COLOR_BGR2GRAY)
                    avg_start = np.mean(start_gray)
                else:
                    avg_start = 0
            else:
                # Fallback if ROI is outside GAME_ROI (shouldn't happen if config is right)
                try:
                    start_screenshot = sct.grab(START_CHECK_ROI)
                    start_img = np.array(start_screenshot)
                    start_gray = cv2.cvtColor(start_img, cv2.COLOR_BGRA2GRAY)
                    avg_start = np.mean(start_gray)
                except:
                    avg_start = 0

            is_playing = avg_start > WHITE_THRESHOLD

            # Check Dead Status (using START_CHECK_ROI and RED_THRESHOLD)
            # We already have start_slice (BGR)
            if 'start_slice' in locals() and start_slice.size > 0:
                # BGR to RGB
                slice_rgb = cv2.cvtColor(start_slice, cv2.COLOR_BGR2RGB)
                avg_r = np.mean(slice_rgb[:, :, 0])
                avg_g = np.mean(slice_rgb[:, :, 1])
                avg_b = np.mean(slice_rgb[:, :, 2])

                is_dead = (avg_r > RED_THRESHOLD['r_min'] and
                           avg_g < RED_THRESHOLD['g_max'] and
                           avg_b < RED_THRESHOLD['b_max'])
            else:
                avg_r, avg_g, avg_b = 0, 0, 0
                is_dead = False

            # Display Status Text
            # Background for text
            # cv2.rectangle(img, (0, 0), (500, 100), (0, 0, 0), -1)

            status_color = (0, 255, 0) if is_playing else (0, 255, 255)
            status_text = f"Start Check: Avg {avg_start:.1f} (>{WHITE_THRESHOLD}) -> {'PLAYING' if is_playing else 'WAITING'}"

            dead_color = (0, 0, 255) if is_dead else (255, 255, 255)
            dead_text = f"Dead Check: R={avg_r:.1f}, G={avg_g:.1f}, B={avg_b:.1f} -> {'DEAD' if is_dead else 'ALIVE'}"
            dead_cond_text = f"Cond: R>{RED_THRESHOLD['r_min']}, G<{RED_THRESHOLD['g_max']}, B<{RED_THRESHOLD['b_max']}"

            cv2.putText(img, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(img, dead_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, dead_color, 2)
            cv2.putText(img, dead_cond_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Env Checker (Press q to quit)", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
