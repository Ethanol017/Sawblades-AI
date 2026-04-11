import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
import pygetwindow as gw
import ctypes
import pymem
import pymem.process

# --- Configuration Constants ---
# 視窗設定
WINDOW_SETTINGS = {
    "title": "A Slight Chance of Sawblades",
    "x": 0,
    "y": 0,
    "width": 800,
    "height": 900,
}

# 遊戲畫面區域 (螢幕座標；假設視窗固定在 WINDOW_SETTINGS x/y)
GAME_ROI = {"top": 101, "left": 170, "width": 460, "height": 626}

# 分數顯示區域
SCORE_ROI = {"top": 101, "left": 310, "width": 200, "height": 70}

# 遊戲開始/結束檢測區域 (畫面中央的小區域)
START_CHECK_ROI = {"top": 101, "left": 170, "width": 460, "height": 200}

# 閾值設定
WHITE_THRESHOLD = 200  # 判定畫面變白的亮度 (Playing)
RED_THRESHOLD = {"r_min": 180, "g_max": 170, "b_max": 170}

# Observation semantic mask settings (RGB color matching)
COLOR_TOLERANCE = 15
PLAYER_COLORS_RGB = [(214, 132, 60), (232, 187, 148)]
OBSTACLE_COLORS_RGB = [(200, 70, 48), (93, 162, 113)]
PLAYER_MASK_VALUE = 255
OBSTACLE_MASK_VALUE = 128

# 獎勵設定
REWARD_SURVIVAL = 0.01
REWARD_SCORE = 1.0
REWARD_POSSIBLE_SCORE = 0 #unable
REWARD_DEATH = -1.0
MAX_RESET_RETRIES = 30

# Score
SCORE_PROCESS_NAME = "SAWBLADES_DEMO.exe"
SCORE_STATIC_OFFSET = 0x00ED7200
SCORE_OFFSETS = [0x20, 0x18, 0x0, 0x8, 0x10, 0x390, 0x0]

# Win32 keyboard message constants (postmessage only)
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
VK_CODES = {
    "left": 0x25,
    "right": 0x27,
    "space": 0x20,
    "enter": 0x0D,
}

# PrintWindow / DIB constants
PW_RENDERFULLCONTENT = 0x00000002
DIB_RGB_COLORS = 0
BI_RGB = 0

_user32 = ctypes.windll.user32
_gdi32 = ctypes.windll.gdi32


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", ctypes.c_uint32 * 1)]


def resize_game_window(activate=True):
    """
    調整遊戲視窗大小與位置。
    此函式為模組級別，方便 tools/check_env.py 等工具調用。
    """
    try:
        windows = gw.getWindowsWithTitle(WINDOW_SETTINGS["title"])
        if windows:
            win = windows[0]
            if win.isMinimized:
                win.restore()
            if activate:
                win.activate()
            win.moveTo(WINDOW_SETTINGS["x"], WINDOW_SETTINGS["y"])
            win.resizeTo(WINDOW_SETTINGS["width"], WINDOW_SETTINGS["height"])
            time.sleep(1)
            print(
                f"Window '{WINDOW_SETTINGS['title']}' resized to {WINDOW_SETTINGS['width']}x{WINDOW_SETTINGS['height']} at ({WINDOW_SETTINGS['x']}, {WINDOW_SETTINGS['y']})"
            )
        else:
            print(f"Window '{WINDOW_SETTINGS['title']}' not found.")
    except Exception as e:
        print(f"Window resize failed: {e}")


class PcGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, auto_resize_window=True, auto_activate_window=False):
        super(PcGameEnv, self).__init__()

        self.game_window = None
        self.game_hwnd = None
        self.capture_origin_x = WINDOW_SETTINGS["x"]
        self.capture_origin_y = WINDOW_SETTINGS["y"]
        self._last_capture_error = ""

        if auto_resize_window:
            resize_game_window(activate=auto_activate_window)
        self._refresh_game_window()

        if self.game_hwnd is None:
            raise RuntimeError("Game window handle not found.")

        # 0: 無操作, 1: 左, 2: 右, 3: 跳, 4: 左+跳, 5: 右+跳
        self.action_space = spaces.Discrete(6)

        # Each frame is uint8 (C,H,W) with two channels: gray + semantic mask.
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(2, 84, 84), dtype=np.uint8
        )

        # 記錄當前按下的鍵，用於狀態機控制
        self.current_keys = {"left": False, "right": False, "space": False}

        # Action Repetition
        self.frame_skip = 4

        # score
        self.pm = None
        self._init_memory_reader()
        self.last_score = 0.0
        self.possible_get_score = False
        self.possible_is_rewarded = False

        self.game_start_time = time.time()

    def _refresh_game_window(self):
        windows = gw.getWindowsWithTitle(WINDOW_SETTINGS["title"])
        if windows:
            self.game_window = windows[0]
            self.game_hwnd = int(getattr(self.game_window, "_hWnd", 0)) or None
            try:
                if (
                    not self.game_window.isMinimized
                    and self.game_window.width > 0
                    and self.game_window.height > 0
                ):
                    self.capture_origin_x = int(self.game_window.left)
                    self.capture_origin_y = int(self.game_window.top)
            except Exception:
                pass
        else:
            self.game_window = None
            self.game_hwnd = None

    def _get_window_capture_size(self):
        """取得本次 PrintWindow 擷取尺寸。"""
        if self.game_window is not None:
            try:
                if self.game_window.width > 0 and self.game_window.height > 0:
                    return int(self.game_window.width), int(self.game_window.height)
            except Exception:
                pass
        return WINDOW_SETTINGS["width"], WINDOW_SETTINGS["height"]

    def _build_capture_debug_message(self, roi=None, stage="unknown", extra=""):
        """建立 capture 失敗時可讀性高的除錯訊息。"""
        self._refresh_game_window()
        window_found = self.game_window is not None
        hwnd = self.game_hwnd

        minimized = None
        left = None
        top = None
        width = None
        height = None
        if self.game_window is not None:
            try:
                minimized = bool(self.game_window.isMinimized)
                left = int(self.game_window.left)
                top = int(self.game_window.top)
                width = int(self.game_window.width)
                height = int(self.game_window.height)
            except Exception:
                pass

        return (
            f"[CaptureDebug] stage={stage}; last_error={self._last_capture_error}; "
            f"window_found={window_found}; hwnd={hwnd}; minimized={minimized}; "
            f"window_rect=({left},{top},{width},{height}); "
            f"capture_origin=({self.capture_origin_x},{self.capture_origin_y}); "
            f"roi={roi}; extra={extra}"
        )

    def _capture_window_bgra(self, width, height):
        """使用 PrintWindow 擷取整個遊戲視窗（BGRA）。"""
        self._last_capture_error = ""
        if self.game_hwnd is None:
            self._refresh_game_window()
            if self.game_hwnd is None:
                self._last_capture_error = "no_window_handle"
                return None

        window_dc = _user32.GetWindowDC(self.game_hwnd)
        if not window_dc:
            self._last_capture_error = "GetWindowDC_failed"
            return None

        mem_dc = None
        hbitmap = None
        old_obj = None

        try:
            mem_dc = _gdi32.CreateCompatibleDC(window_dc)
            if not mem_dc:
                self._last_capture_error = "CreateCompatibleDC_failed"
                return None

            hbitmap = _gdi32.CreateCompatibleBitmap(window_dc, width, height)
            if not hbitmap:
                self._last_capture_error = "CreateCompatibleBitmap_failed"
                return None

            old_obj = _gdi32.SelectObject(mem_dc, hbitmap)

            ok = _user32.PrintWindow(self.game_hwnd, mem_dc, PW_RENDERFULLCONTENT)
            if not ok:
                ok = _user32.PrintWindow(self.game_hwnd, mem_dc, 0)
            if not ok:
                self._last_capture_error = "PrintWindow_failed"
                return None

            bmi = BITMAPINFO()
            bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.bmiHeader.biWidth = width
            bmi.bmiHeader.biHeight = -height  # top-down
            bmi.bmiHeader.biPlanes = 1
            bmi.bmiHeader.biBitCount = 32
            bmi.bmiHeader.biCompression = BI_RGB

            buffer = ctypes.create_string_buffer(width * height * 4)
            lines = _gdi32.GetDIBits(
                mem_dc,
                hbitmap,
                0,
                height,
                buffer,
                ctypes.byref(bmi),
                DIB_RGB_COLORS,
            )
            if lines == 0:
                self._last_capture_error = "GetDIBits_failed"
                return None

            img = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
            return img.copy()

        except Exception as e:
            self._last_capture_error = f"capture_exception:{e}"
            return None

        finally:
            if old_obj and mem_dc:
                _gdi32.SelectObject(mem_dc, old_obj)
            if hbitmap:
                _gdi32.DeleteObject(hbitmap)
            if mem_dc:
                _gdi32.DeleteDC(mem_dc)
            _user32.ReleaseDC(self.game_hwnd, window_dc)

    def _crop_window_roi(self, window_img, roi):
        """把螢幕座標 ROI 轉成視窗座標後裁切；超界時直接丟錯。"""
        rel_left = int(roi["left"] - self.capture_origin_x)
        rel_top = int(roi["top"] - self.capture_origin_y)
        target_w = int(roi["width"])
        target_h = int(roi["height"])

        if target_w <= 0 or target_h <= 0:
            raise RuntimeError(
                self._build_capture_debug_message(
                    roi=roi,
                    stage="invalid_roi_size",
                    extra=f"target_w={target_w},target_h={target_h}",
                )
            )

        x0 = max(0, rel_left)
        y0 = max(0, rel_top)
        x1 = min(window_img.shape[1], rel_left + target_w)
        y1 = min(window_img.shape[0], rel_top + target_h)

        if x1 <= x0 or y1 <= y0:
            raise RuntimeError(
                self._build_capture_debug_message(
                    roi=roi,
                    stage="roi_out_of_window",
                    extra=(
                        f"rel_left={rel_left},rel_top={rel_top},"
                        f"target_w={target_w},target_h={target_h},"
                        f"window_shape={window_img.shape}"
                    ),
                )
            )

        if (x1 - x0) != target_w or (y1 - y0) != target_h:
            raise RuntimeError(
                self._build_capture_debug_message(
                    roi=roi,
                    stage="partial_roi_intersection",
                    extra=(
                        f"requested=({target_w},{target_h}),"
                        f"actual=({x1 - x0},{y1 - y0}),"
                        f"window_shape={window_img.shape}"
                    ),
                )
            )

        return window_img[y0:y1, x0:x1].copy()

    def _grab(self, roi):
        self._refresh_game_window()
        cap_w, cap_h = self._get_window_capture_size()
        window_img = self._capture_window_bgra(cap_w, cap_h)
        if window_img is None:
            raise RuntimeError(
                self._build_capture_debug_message(
                    roi=roi,
                    stage="capture_failed",
                    extra=f"capture_size=({cap_w},{cap_h})",
                )
            )
        return self._crop_window_roi(window_img, roi)

    def _post_message_key(self, key, is_key_down):
        if self.game_hwnd is None:
            self._refresh_game_window()
            if self.game_hwnd is None:
                return

        vk = VK_CODES[key]
        scan_code = _user32.MapVirtualKeyW(vk, 0)
        lparam = 1 | (scan_code << 16)
        msg = WM_KEYDOWN
        if not is_key_down:
            msg = WM_KEYUP
            lparam |= 1 << 30
            lparam |= 1 << 31
        _user32.PostMessageW(self.game_hwnd, msg, vk, lparam)

    def _key_down(self, key):
        self._post_message_key(key, is_key_down=True)

    def _key_up(self, key):
        self._post_message_key(key, is_key_down=False)

    def _press_key(self, key):
        self._post_message_key(key, is_key_down=True)
        self._post_message_key(key, is_key_down=False)

    def _color_mask(self, rgb_img, target_rgb, tolerance):
        target = np.array(target_rgb, dtype=np.int16)
        lower = np.clip(target - tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(target + tolerance, 0, 255).astype(np.uint8)
        return cv2.inRange(rgb_img, lower, upper)

    def _build_semantic_mask(self, rgb_img):
        h, w = rgb_img.shape[:2]
        player_mask = np.zeros((h, w), dtype=np.uint8)
        obstacle_mask = np.zeros((h, w), dtype=np.uint8)

        for color in PLAYER_COLORS_RGB:
            player_mask = cv2.bitwise_or(
                player_mask, self._color_mask(rgb_img, color, COLOR_TOLERANCE)
            )

        for color in OBSTACLE_COLORS_RGB:
            obstacle_mask = cv2.bitwise_or(
                obstacle_mask, self._color_mask(rgb_img, color, COLOR_TOLERANCE)
            )

        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        semantic_mask[obstacle_mask > 0] = OBSTACLE_MASK_VALUE
        # Player assignment comes last so the player stays highlighted on overlap.
        semantic_mask[player_mask > 0] = PLAYER_MASK_VALUE
        return semantic_mask

    def _process_obs(self, sct_img):
        """
        處理觀察值: 轉灰階(反轉) + 顏色語意遮罩 -> Resize (84x84) -> 回傳 uint8 (2,84,84)
        """
        img = np.array(sct_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        gray = 255 - gray  # 反轉顏色：遊戲中是白底黑圖，反轉後變黑底白圖更適合 CNN 學習邊緣等特徵。
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        semantic_mask = self._build_semantic_mask(rgb)

        gray_resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(
            semantic_mask, (84, 84), interpolation=cv2.INTER_NEAREST
        )
        obs = np.stack([gray_resized, mask_resized], axis=0)
        return obs.astype(np.uint8)

    def _check_game_status(self, sct_img):
        """
        檢查遊戲狀態 (Playing/Dead/Unknown)
        以及檢查是否有可能得分 (possible_get_score)
        """
        if time.time() - self.game_start_time > 32.0:
            return "Timeout"
        status = "Unknown"
        img = np.array(sct_img)

        # 從 GAME_ROI 切出 START_CHECK_ROI
        rel_top = START_CHECK_ROI["top"] - GAME_ROI["top"]
        rel_left = START_CHECK_ROI["left"] - GAME_ROI["left"]

        # possible get score
        _rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        _target = np.array([93, 162, 113], dtype=np.uint8)
        _tol = 0
        _lower = np.clip(_target - _tol, 0, 255).astype(np.uint8)
        _upper = np.clip(_target + _tol, 0, 255).astype(np.uint8)
        _mask = cv2.inRange(_rgb, _lower, _upper)
        self.possible_get_score = bool(np.any(_mask))

        if not self.possible_get_score:
            self.possible_is_rewarded = False

        start_slice = img[
            rel_top : rel_top + START_CHECK_ROI["height"],
            rel_left : rel_left + START_CHECK_ROI["width"],
        ]

        if start_slice.size == 0:
            return status

        # 檢測白色 (Playing)
        start_gray = cv2.cvtColor(start_slice, cv2.COLOR_BGRA2GRAY)
        avg_start = np.mean(start_gray)

        # 檢測紅色 (Dead)
        slice_rgb = cv2.cvtColor(start_slice, cv2.COLOR_BGRA2RGB)
        avg_r = np.mean(slice_rgb[:, :, 0])
        avg_g = np.mean(slice_rgb[:, :, 1])
        avg_b = np.mean(slice_rgb[:, :, 2])

        if (
            avg_r > RED_THRESHOLD["r_min"]
            and avg_g < RED_THRESHOLD["g_max"]
            and avg_b < RED_THRESHOLD["b_max"]
        ):
            status = "Dead"
        elif avg_start > WHITE_THRESHOLD:
            status = "Playing"

        return status

    def _apply_action(self, action):
        """將原本 step 中的按鍵邏輯抽離出來"""
        need_left = False
        need_right = False
        need_jump = False

        if action == 1:  # 左
            need_left = True
        elif action == 2:  # 右
            need_right = True
        elif action == 3:  # 跳
            need_jump = True
        elif action == 4:  # 左 + 跳
            need_left = True
            need_jump = True
        elif action == 5:  # 右 + 跳
            need_right = True
            need_jump = True

        if need_left and not self.current_keys["left"]:
            self._key_down("left")
            self.current_keys["left"] = True
        elif not need_left and self.current_keys["left"]:
            self._key_up("left")
            self.current_keys["left"] = False

        if need_right and not self.current_keys["right"]:
            self._key_down("right")
            self.current_keys["right"] = True
        elif not need_right and self.current_keys["right"]:
            self._key_up("right")
            self.current_keys["right"] = False

        if need_jump and not self.current_keys["space"]:
            self._key_down("space")
            self.current_keys["space"] = True
        elif not need_jump and self.current_keys["space"]:
            self._key_up("space")
            self.current_keys["space"] = False

    def _init_memory_reader(self):
        """初始化 pymem 並設定指針路徑"""
        try:
            self.pm = pymem.Pymem(SCORE_PROCESS_NAME)
            self.game_module = pymem.process.module_from_name(
                self.pm.process_handle, SCORE_PROCESS_NAME
            ).lpBaseOfDll
        except Exception as e:
            print(f"記憶體讀取初始化失敗: {e}")
            self.pm = None

    def get_score_from_memory(self):
        """從記憶體讀取當前分數"""
        if self.pm is None:
            return 0
        try:
            addr = self.game_module + SCORE_STATIC_OFFSET
            for offset in SCORE_OFFSETS:
                addr = self.pm.read_longlong(addr)
                if addr == 0:
                    return 0
                addr = addr + offset
            return self.pm.read_double(addr)
        except Exception:
            return 0

    def _calculate_reward(self, status):
        score_reward = 0.0
        score = self.get_score_from_memory()
        if score > self.last_score:
            score_reward = (score - self.last_score) * REWARD_SCORE
            self.last_score = score

        if self.possible_get_score and not self.possible_is_rewarded:
            score_reward += REWARD_POSSIBLE_SCORE
            self.possible_is_rewarded = True

        reward = score_reward
        if status == "Dead":
            reward += REWARD_DEATH

        return reward

    def step(self, action):
        total_reward = REWARD_SURVIVAL
        terminated = False
        truncated = False

        self._apply_action(action)

        for frame_idx in range(self.frame_skip):
            try:
                sct_img = self._grab(GAME_ROI)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Capture failed in step(action={action}, frame_idx={frame_idx}): {e}"
                )

            status = self._check_game_status(sct_img)
            terminated = status == "Dead"
            truncated = status == "Timeout"
            # if terminated or truncated:
            # print(time.time() - self.game_start_time)
            # print(f"Game ended with status: {status}")
            # img = np.array(sct_img)
            # cv2.imwrite(f"debug_picture/debug_game_over{int(time.time())}.png", img)

            step_reward = self._calculate_reward(status)
            total_reward += step_reward

            time.sleep(0.004)

            if terminated or truncated:
                break

        obs = self._process_obs(sct_img)
        return obs, total_reward, terminated, truncated, {}

    def _release_all_keys(self):
        self._key_up("left")
        self._key_up("right")
        self._key_up("space")
        self.current_keys = {"left": False, "right": False, "space": False}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_score = 0.0

        self._release_all_keys()

        time.sleep(2.5)
        self._press_key("enter")

        self.game_start_time = time.time()
        retry_count = 0

        while True:
            try:
                sct_img = self._grab(GAME_ROI)
            except RuntimeError as e:
                raise RuntimeError(f"Capture failed during reset loop: {e}")

            status = self._check_game_status(sct_img)

            if status == "Playing":
                break

            if time.time() - self.game_start_time > 1.0:
                retry_count += 1
                print(
                    f"Reset timeout (Status: {status}), retrying Enter... ({retry_count}/{MAX_RESET_RETRIES})"
                )
                if retry_count >= MAX_RESET_RETRIES:
                    raise RuntimeError(
                        self._build_capture_debug_message(
                            roi=GAME_ROI,
                            stage="reset_retry_exceeded",
                            extra=f"last_status={status},retries={retry_count}",
                        )
                    )
                self._press_key("enter")
                self.game_start_time = time.time()
                time.sleep(0.5)

            time.sleep(0.1)

        try:
            sct_img = self._grab(GAME_ROI)
        except RuntimeError as e:
            raise RuntimeError(f"Capture failed after reset success: {e}")

        current_frame = self._process_obs(sct_img)
        return current_frame, {}

    def close(self):
        try:
            self._release_all_keys()
        except Exception:
            pass


# Just for quick testing the environment with random actions.
if __name__ == "__main__":
    env = PcGameEnv()
    obs, info = env.reset()
    for _ in range(400):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # save obs to image file for debugging; only need one image.
        debug_img = np.concatenate([obs[0], obs[1]], axis=1)
        cv2.imwrite("debug_picture/debug_obs.png", debug_img)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
