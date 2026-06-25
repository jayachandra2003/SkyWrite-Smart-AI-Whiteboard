import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from PIL import Image, ImageDraw, ImageFont
import os
import time
import threading
import math
from collections import deque

# --- 1. SETUP & CONFIGURATION ---
MODEL_PATH = 'my_emnist_model.h5' 
FONT_PATH = "font.ttf"
FONT_SIZE = 50
LOAD_FILE_NAME = "load_canvas.png"  

# --- 2. HELPER FUNCTIONS & CLASSES ---

def get_or_create_model():
    """Safely builds network structure and maps the weight values to bypass Keras version mismatches."""
    if os.path.exists(MODEL_PATH):
        print("Reconstructing model layer structure to align weights file...")
        try:
            model = Sequential([
                tf.keras.layers.Input(shape=(28, 28, 1), name='input_layer_fixed'), 
                Conv2D(32, kernel_size=(3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(64, kernel_size=(3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dropout(0.5),
                Dense(62, activation='softmax')
            ])
            model.load_weights(MODEL_PATH)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            print("Locally saved EMNIST AI model weights loaded successfully!")
            return model
        except Exception as e:
            print(f"AI STRUCTURE ERROR: Failed to map weights vector matrix. {e}")
            return None
    else:
        print(f"AI ERROR: Model weights file '{MODEL_PATH}' was not found!")
        return None

class Button:
    """Manages an individual Responsive UI toolbar item with pure color & gradient support."""
    def __init__(self, key_id, base_x, base_w, text, color=(240, 240, 240), is_color_preset=False):
        self.key_id = key_id
        self.base_x = base_x
        self.base_w = base_w
        self.text = text
        self.color = color
        self.is_color_preset = is_color_preset
        self.last_click_time = 0

    def draw(self, image, scale_factor, highlight=False, hovered=False, text_color=(20, 20, 20)):
        gap = int(8 * scale_factor)
        y = gap
        h = int(50 * scale_factor)
        x = int(self.base_x * scale_factor)
        w = int(self.base_w * scale_factor)
        
        if self.is_color_preset:
            cv2.rectangle(image, (x, y), (x + w, y + h), self.color, -1)
        elif self.text == "Spectrum":
            for i in range(w):
                hue = int((i / w) * 179)
                hsv_pixel = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                bgr_col = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
                cv2.line(image, (x + i, y), (x + i, y + h), tuple(int(c) for c in bgr_col), 1)
        else:
            overlay = image.copy()
            bg_color = (130, 250, 130) if highlight else ((180, 220, 255) if hovered else self.color)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
            alpha = 0.55 if hovered or highlight else 0.35
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
        
        border_color = (0, 200, 0) if highlight else ((255, 120, 0) if hovered else (160, 160, 160))
        border_thickness = 2 if highlight or hovered else 1
        cv2.rectangle(image, (x, y), (x + w, y + h), border_color, border_thickness)

        if self.text:
            font_scale = 0.42 * scale_factor
            thickness = max(1, int(1 * scale_factor))
            text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            
            if self.text == "Spectrum":
                cv2.putText(image, self.text, (text_x + 1, text_y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (10, 10, 10), thickness, cv2.LINE_AA)
                cv2.putText(image, self.text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            else:
                cv2.putText(image, self.text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
        return image

    def is_clicked(self, point, scale_factor):
        # --- FIXED FEATURE: Dynamic calculations based on active scaling to prevent cross-window unresponsiveness ---
        gap = int(8 * scale_factor)
        y = gap
        h = int(50 * scale_factor)
        x = int(self.base_x * scale_factor)
        w = int(self.base_w * scale_factor)
        
        current_time_ms = time.time() * 1000
        if x < point[0] < x + w and y < point[1] < y + h and (current_time_ms - self.last_click_time) > 600: 
            self.last_click_time = current_time_ms
            return True
        return False

# --- 3. MAIN APPLICATION CLASS ---

class AirCanvas:
    def __init__(self):
        self.model, self.font = None, None
        try:
             self.model = get_or_create_model()
             self.font = ImageFont.truetype(FONT_PATH, FONT_SIZE) if os.path.exists(FONT_PATH) else ImageFont.load_default()
        except Exception as e: 
             print(f"Error loading resources: {e}")

        self.label_mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened(): 
             print("FATAL: Cannot open camera.")
             return

        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.screen_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.screen_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.header_height = 80
        self.min_thickness = 2
        self.max_thickness = 50

        # DARK MODE / LIGHT MODE PROPERTIES
        self.dark_mode_active = False
        self.bg_color_val = 255
        self.grid_color_val = (230, 230, 230)
        self.ink_color_val = (30, 30, 30)

        # Base Window Layers
        self.paint_window = np.full((self.screen_height, self.screen_width, 3), self.bg_color_val, dtype=np.uint8)
        
        # Load local template background if active
        self.handle_external_file_loader()
        
        # Undo/Redo registry arrays stacks configuration
        self.undo_stack = deque(maxlen=20)
        self.redo_stack = deque(maxlen=20)
        self.save_canvas_snapshot()

        # Core operational tracking parameters
        self.current_color = (0, 0, 255) 
        self.current_mode = "DRAW"
        self.pre_pinch_mode = "DRAW"  
        self.auto_write_mode = False  
        self.thickness = 8 
        
        # SLIDER SUB-PANEL PROPERTIES
        self.slider_visible = False
        self.slider_x = 420
        self.slider_y = 100
        self.slider_width = 250
        self.slider_height = 25

        # ADVANCED CONTINUOUS COLOR WHEEL MESH PROPERTIES
        self.wheel_visible = False
        self.wheel_center = (640, 280)  
        self.wheel_radius = 140         
        self.generate_spectrum_mesh_cache()

        self.is_drawing = False
        self.shape_start_point = None
        self.last_point = (0, 0)
        self.point_buffer = deque(maxlen=4)
        self.letter_points = []  

        self.swipe_history = deque(maxlen=8)
        self.setup_ui_layout()
        self._reset_text_cursor()

        # Initialize windows explicitly with normal flags and keep ratio locked
        cv2.namedWindow("SkyWrite Dashboard", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Whiteboard Canvas", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("SkyWrite Dashboard", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty("Whiteboard Canvas", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

        # Voice thread configuration properties 
        self.voice_command_string = "Listening for audio words..."
        self.start_voice_async_thread()

    def generate_spectrum_mesh_cache(self):
        """Pre-renders a high-fidelity HSV circular spectrum wheel buffer."""
        size = self.wheel_radius * 2
        self.wheel_cache = np.zeros((size, size, 3), dtype=np.uint8)
        for y in range(size):
            for x in range(size):
                dx = x - self.wheel_radius
                dy = y - self.wheel_radius
                distance = math.sqrt(dx*dx + dy*dy)
                if distance <= self.wheel_radius:
                    angle = math.atan2(dy, dx)
                    hue = int(((angle + math.pi) / (2 * math.pi)) * 179)
                    saturation = int((distance / self.wheel_radius) * 255)
                    value = 255
                    
                    hsv_pixel = np.array([[[hue, saturation, value]]], dtype=np.uint8)
                    self.wheel_cache[y, x] = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]

    def handle_external_file_loader(self):
        """Loads workspace template maps directly from local file directories to resume tasks."""
        if os.path.exists(LOAD_FILE_NAME):
            print(f"External Template Detected: Parsing '{LOAD_FILE_NAME}' onto whiteboard grid sheets...")
            try:
                loaded_img = cv2.imread(LOAD_FILE_NAME)
                resized_img = cv2.resize(loaded_img, (self.screen_width, self.screen_height))
                self.paint_window = resized_img
            except Exception as e:
                print(f"File Load Error: Core configurations corrupted. {e}")
                self.generate_grid_notebook()
        else:
            self.generate_grid_notebook()

    def generate_grid_notebook(self):
        """Draws a notebook paper structure background overlay matching workspace brightness constraints."""
        for y in range(self.header_height + 30, self.screen_height, 30):
            for x in range(30, self.screen_width, 30):
                cv2.circle(self.paint_window, (x, y), 1, self.grid_color_val, -1)

    def toggle_theme_workspace(self):
        """Swaps layout design settings between light and obsidian palettes dynamically."""
        self.dark_mode_active = not self.dark_mode_active
        
        old_bg = self.bg_color_val
        if self.dark_mode_active:
            self.bg_color_val = 15  
            self.grid_color_val = (45, 45, 45)
            self.ink_color_val = (240, 240, 240)
        else:
            self.bg_color_val = 255 
            self.grid_color_val = (230, 230, 230)
            self.ink_color_val = (30, 30, 30)

        gray_canvas = cv2.cvtColor(self.paint_window, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, old_bg - 5 if old_bg > 128 else old_bg + 5, 255, cv2.THRESH_BINARY)
        if old_bg == 255:
            mask = cv2.bitwise_not(mask)

        new_base = np.full(self.paint_window.shape, self.bg_color_val, dtype=np.uint8)
        self.generate_grid_notebook()
        
        idx = (mask == 255)
        new_base[idx] = self.paint_window[idx]
        
        if self.dark_mode_active:
            new_base[np.all(new_base == (30,30,30), axis=-1)] = (240,240,240)
        else:
            new_base[np.all(new_base == (240,240,240), axis=-1)] = (30,30,30)

        self.paint_window = new_base
        self.save_canvas_snapshot()

    def save_canvas_snapshot(self):
        self.undo_stack.append(self.paint_window.copy())

    def trigger_undo(self):
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop()) 
            self.paint_window = self.undo_stack[-1].copy()
            print("Action Undo performed.")
        else:
            print("Undo stack frame bound hit.")

    def trigger_redo(self):
        if self.redo_stack:
            state_frame = self.redo_stack.pop()
            self.undo_stack.append(state_frame)
            self.paint_window = state_frame.copy()
            print("Action Redo performed.")
        else:
            print("Redo cache empty.")

    def start_voice_async_thread(self):
        """Spins up a separate background voice listener to process audio speech inputs safely."""
        try:
            import speech_recognition as sr
            def voice_listener_loop():
                recognizer = sr.Recognizer()
                microphone = sr.Microphone()
                with microphone as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                
                while self.cap.isOpened():
                    with microphone as source:
                        try:
                            audio = recognizer.listen(source, phrase_time_limit=2)
                            command = recognizer.recognize_google(audio).lower()
                            
                            command = command.replace("colour", "color")
                            self.voice_command_string = f"Voice Command Heard: '{command}'"
                            
                            if "clear canvas" in command or "wipe screen" in command:
                                self.paint_window.fill(self.bg_color_val)
                                self.generate_grid_notebook()
                                self.save_canvas_snapshot()
                            elif "color red" in command:
                                self.current_color = (0, 0, 255)
                                self.current_mode = "DRAW"
                                self.pre_pinch_mode = "DRAW"
                            elif "color blue" in command:
                                self.current_color = (255, 0, 0)
                                self.current_mode = "DRAW"
                                self.pre_pinch_mode = "DRAW"
                            elif "color green" in command:
                                self.current_color = (0, 255, 0)
                                self.current_mode = "DRAW"
                                self.pre_pinch_mode = "DRAW"
                            elif "color black" in command:
                                self.current_color = (0, 0, 0) if not self.dark_mode_active else (255,255,255)
                                self.current_mode = "DRAW"
                                self.pre_pinch_mode = "DRAW"
                            elif "mode dark" in command and not self.dark_mode_active:
                                self.toggle_theme_workspace()
                            elif "mode light" in command and self.dark_mode_active:
                                self.toggle_theme_workspace()
                            elif "action undo" in command:
                                self.trigger_undo()
                        except sr.WaitTimeoutError: pass
                        except sr.UnknownValueError: pass
                        except Exception: pass
                        
            t = threading.Thread(target=voice_listener_loop, daemon=True)
            t.start()
        except Exception as e:
            self.voice_command_string = "Voice Engine Offline: speech_recognition not found."
            print(f"Voice processor skipped: {e}")

    def setup_ui_layout(self):
        btn_w, gap = 60, 8
        self.buttons = []
        
        modes = ["Pen", "Rect", "Circ", "Cube", "Fill", "Erase"]
        keys = ["DRAW", "RECT", "CIRCLE", "CUBE", "FILL", "ERASER"]
        for i, text in enumerate(modes):
            cx = gap + i * (btn_w + gap)
            self.buttons.append(Button(keys[i], cx, btn_w, text))
            
        cx = gap + 6 * (btn_w + gap) + 10
        self.buttons.append(Button("SIZE_P", cx, 75, "Pen Size"))
        self.buttons.append(Button("THEME", cx + 75 + gap, btn_w, "Theme"))
        
        cx += 75 + btn_w + 2 * gap + 15
        self.colors_list = [
            ("RED", (0,0,255)), ("ORANGE", (0,165,255)), ("YELLOW", (0,255,255)),
            ("GREEN", (0,255,0)), ("BLUE", (255,0,0)), ("PURPLE", (128,0,128))
        ]
        for name, bgr in self.colors_list:
            self.buttons.append(Button(name, cx, 30, "", color=bgr, is_color_preset=True))
            cx += (30 + gap)

        self.buttons.append(Button("WHEEL_T", cx, 80, "Spectrum"))
        
        rx = self.screen_width - (5 * (btn_w + gap)) - gap
        self.buttons.append(Button("AUTO-W", rx, btn_w, "AI Mode"))
        self.buttons.append(Button("UNDO", rx + 1 * (btn_w + gap), btn_w, "Undo"))
        self.buttons.append(Button("REDO", rx + 2 * (btn_w + gap), btn_w, "Redo"))
        self.buttons.append(Button("CLEAR", rx + 3 * (btn_w + gap), btn_w, "Clear"))
        self.buttons.append(Button("SAVE", rx + 4 * (btn_w + gap), btn_w, "Save"))

    def _reset_text_cursor(self):
        self.text_margin = 40
        self.line_height = FONT_SIZE + 15
        self.text_cursor_x = self.text_margin
        self.text_cursor_y = self.header_height + 40

    def draw_unique_skeleton(self, lms, canvas_target, is_whiteboard=False):
        """Renders an advanced bone mesh structure matching dark/light contrasts cleanly."""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (5, 9), (9, 10), (10, 11), (11, 12), # Middle
            (9, 13), (13, 14), (14, 15), (15, 16),# Ring
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
        ]
        
        bone_color = (180, 180, 180) if is_whiteboard else (230, 245, 255)
        for connection in connections:
            pt1 = lms[connection[0]]
            pt2 = lms[connection[1]]
            cv2.line(canvas_target, pt1, pt2, bone_color, 1, cv2.LINE_AA)
            
        for idx, pt in enumerate(lms):
            if idx in [4, 8, 12, 16, 20]:
                cv2.circle(canvas_target, pt, 5, (255, 120, 0) if is_whiteboard else (0, 215, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas_target, pt, 8, (255, 200, 150) if is_whiteboard else (255, 150, 100), 1, cv2.LINE_AA)
            else:
                cv2.circle(canvas_target, pt, 3, (100, 100, 100) if is_whiteboard else (255, 255, 255), -1, cv2.LINE_AA)

    def draw_slider_subpanel(self, target_image, scale_factor, text_color):
        if not self.slider_visible: return
        
        sx = int(self.slider_x * scale_factor)
        sy = int(self.slider_y * scale_factor)
        sw = int(self.slider_width * scale_factor)
        sh = int(self.slider_height * scale_factor)
        
        overlay = target_image.copy()
        cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), (220, 220, 220), -1)
        cv2.addWeighted(overlay, 0.4, target_image, 0.6, 0, dst=target_image)
        cv2.rectangle(target_image, (sx, sy), (sx + sw, sy + sh), (160, 160, 160), 1, cv2.LINE_AA)
        
        percentage = (self.thickness - self.min_thickness) / (self.max_thickness - self.min_thickness)
        filled_w = int(percentage * sw)
        
        cv2.rectangle(target_image, (sx, sy), (sx + filled_w, sy + sh), (0, 165, 255), -1)
        
        knob_cx = sx + filled_w
        knob_cy = sy + (sh // 2)
        cv2.circle(target_image, (knob_cx, knob_cy), int(8 * scale_factor), (0, 120, 255), -1, cv2.LINE_AA)
        cv2.circle(target_image, (knob_cx, knob_cy), int(9 * scale_factor), (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(target_image, f"Width: {self.thickness}px", (sx + sw + int(15 * scale_factor), sy + int(17 * scale_factor)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale_factor, text_color, 1, cv2.LINE_AA)

    def check_slider_hover_events(self, pointer, scale_factor):
        if not self.slider_visible: return False
        
        sx = int(self.slider_x * scale_factor)
        sy = int(self.slider_y * scale_factor)
        sw = int(self.slider_width * scale_factor)
        sh = int(self.slider_height * scale_factor)
        
        px, py = pointer[0], pointer[1]
        if sx <= px <= (sx + sw) and sy <= py <= (sy + sh):
            ratio = (px - sx) / sw
            self.thickness = int(ratio * (self.max_thickness - self.min_thickness) + self.min_thickness)
            return True
        return False

    def draw_pro_spectrum_wheel(self, target_image, scale_factor, text_color):
        if not self.wheel_visible: return
        
        wx = int(self.wheel_center[0] * scale_factor)
        wy = int(self.wheel_center[1] * scale_factor)
        r = int(self.wheel_radius * scale_factor)
        
        if r <= 5: return
        
        wheel_resized = cv2.resize(self.wheel_cache, (r * 2, r * 2), interpolation=cv2.INTER_AREA)
        
        roi = target_image[wy-r:wy+r, wx-r:wx+r]
        mask = np.zeros((r * 2, r * 2), dtype=np.uint8)
        cv2.circle(mask, (r, r), r, 255, -1)
        
        idx = (mask == 255)
        roi[idx] = cv2.addWeighted(wheel_resized, 0.95, roi, 0.05, 0)[idx]
        
        cv2.circle(target_image, (wx, wy), r + int(6 * scale_factor), self.current_color, int(8 * scale_factor), cv2.LINE_AA)
        cv2.circle(target_image, (wx, wy), r + int(11 * scale_factor), (160, 160, 160), 1, cv2.LINE_AA)
        cv2.circle(target_image, (wx, wy), r, (255, 255, 255), 2, cv2.LINE_AA)
        
        hex_string = f"HEX: #{self.current_color[2]:02X}{self.current_color[1]:02X}{self.current_color[0]:02X}"
        cv2.rectangle(target_image, (wx - int(65 * scale_factor), wy + r + int(20 * scale_factor)), 
                      (wx + int(65 * scale_factor), wy + r + int(42 * scale_factor)), (30,30,30) if not self.dark_mode_active else (230,230,230), -1)
        cv2.putText(target_image, hex_string, (wx - int(52 * scale_factor), wy + r + int(36 * scale_factor)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale_factor, (240,240,240) if not self.dark_mode_active else (20,20,20), 1, cv2.LINE_AA)

    def sample_wheel_color(self, pointer, scale_factor):
        if not self.wheel_visible: return False
        
        wx = int(self.wheel_center[0] * scale_factor)
        wy = int(self.wheel_center[1] * scale_factor)
        r = int(self.wheel_radius * scale_factor)
        
        px, py = pointer[0], pointer[1]
        dx = px - wx
        dy = py - wy
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance <= r and r > 0:
            angle = math.atan2(dy, dx)
            hue = int(((angle + math.pi) / (2 * math.pi)) * 179)
            saturation = int((distance / r) * 255)
            value = 255
            
            hsv_pixel = np.array([[[hue, saturation, value]]], dtype=np.uint8)
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
            
            self.current_color = tuple(int(c) for c in bgr_pixel)
            return True
        return False

    def get_window_scale_factor(self, window_name):
        try:
            rect = cv2.getWindowImageRect(window_name)
            if rect and rect[2] > 100:
                return float(rect[2] / self.screen_width)
        except Exception:
            pass
        return 1.0

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(framergb)
            
            scale_dash = self.get_window_scale_factor("SkyWrite Dashboard")
            scale_canvas = self.get_window_scale_factor("Whiteboard Canvas")

            output_frame = cv2.resize(frame, (int(self.screen_width * scale_dash), int(self.screen_height * scale_dash)), interpolation=cv2.INTER_AREA)
            whiteboard_scratchpad = cv2.resize(self.paint_window, (int(self.screen_width * scale_canvas), int(self.screen_height * scale_canvas)), interpolation=cv2.INTER_AREA)
            
            for layer, sf in [(output_frame, scale_dash), (whiteboard_scratchpad, scale_canvas)]:
                scaled_h = int(self.header_height * sf)
                scaled_w = layer.shape[1]
                if scaled_h > 0 and scaled_w > 0:
                    header_roi = layer[0:scaled_h, 0:scaled_w]
                    layer[0:scaled_h, 0:scaled_w] = cv2.GaussianBlur(header_roi, (25, 25), 0)
                    cv2.rectangle(layer, (0, 0), (scaled_w, scaled_h), (15,15,15) if self.dark_mode_active else (255, 255, 255), 1)

            if result.multi_hand_landmarks:
                lms_dash = [(int(lm.x * output_frame.shape[1]), int(lm.y * output_frame.shape[0])) for lm in result.multi_hand_landmarks[0].landmark]
                lms_canvas = [(int(lm.x * whiteboard_scratchpad.shape[1]), int(lm.y * whiteboard_scratchpad.shape[0])) for lm in result.multi_hand_landmarks[0].landmark]
                
                self.process_frame_hand_vectors(lms_dash, scale_dash)
                
                self.draw_unique_skeleton(lms_dash, output_frame, is_whiteboard=False)
                self.draw_unique_skeleton(lms_canvas, whiteboard_scratchpad, is_whiteboard=True)
            else:
                if self.is_drawing: self.finalize_action()
                self.is_drawing = False
                self.last_point = (0,0)
                self.point_buffer.clear()

            txt_bgr_lbl = (230, 230, 230) if self.dark_mode_active else (30, 30, 30)
            whiteboard_txt_lbl = (240, 240, 240) if self.dark_mode_active else (20, 20, 20)
            
            output_frame = self.draw_all_ui(output_frame, scale_dash, txt_bgr_lbl)
            whiteboard_scratchpad = self.draw_all_ui(whiteboard_scratchpad, scale_canvas, whiteboard_txt_lbl)
            
            self.draw_slider_subpanel(output_frame, scale_dash, txt_bgr_lbl)
            self.draw_slider_subpanel(whiteboard_scratchpad, scale_canvas, whiteboard_txt_lbl)
            
            self.draw_pro_spectrum_wheel(output_frame, scale_dash, txt_bgr_lbl)
            self.draw_pro_spectrum_wheel(whiteboard_scratchpad, scale_canvas, whiteboard_txt_lbl)
            
            status_text = f"Mode: {self.current_mode} | Size: {self.thickness}px | {self.voice_command_string}"
            cv2.putText(output_frame, status_text, (int(20 * scale_dash), output_frame.shape[0] - int(20 * scale_dash)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale_dash, txt_bgr_lbl, 1, cv2.LINE_AA)
            cv2.putText(whiteboard_scratchpad, status_text, (int(20 * scale_canvas), whiteboard_scratchpad.shape[0] - int(20 * scale_canvas)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale_canvas, whiteboard_txt_lbl, 1, cv2.LINE_AA)

            paint_resized = cv2.resize(self.paint_window, (output_frame.shape[1], output_frame.shape[0]), interpolation=cv2.INTER_AREA)
            final_frame = cv2.addWeighted(output_frame, 0.75, paint_resized, 0.25, 0)
            
            cv2.imshow("SkyWrite Dashboard", final_frame)
            cv2.imshow("Whiteboard Canvas", whiteboard_scratchpad)

            if cv2.waitKey(1) == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()

    def process_frame_hand_vectors(self, lms, scale_factor):
        thumb_tip, index_tip, middle_tip, pinky_tip = lms[4], lms[8], lms[12], lms[20]
        palm_center = lms[9]

        is_index_up = index_tip[1] < lms[6][1]
        is_middle_up = middle_tip[1] < lms[10][1]
        is_pinky_up = pinky_tip[1] < lms[18][1]

        self.swipe_history.append(palm_center[0])
        if len(self.swipe_history) == self.swipe_history.maxlen:
            h_velocity = self.swipe_history[-1] - self.swipe_history[0]
            if h_velocity < -320 and is_pinky_up:
                self.paint_window.fill(self.bg_color_val)
                self.generate_grid_notebook()
                self.save_canvas_snapshot()
                self._reset_text_cursor()
                self.swipe_history.clear()
                print("Canvas Wiped.")
                return

        distance_pinch = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
        
        if distance_pinch < int(25 * scale_factor) and not is_middle_up:
            if self.current_mode != "ERASER":
                self.pre_pinch_mode = self.current_mode  
                self.current_mode = "ERASER"
            
            base_index = (int(index_tip[0] / scale_factor), int(index_tip[1] / scale_factor))
            if self.last_point != (0,0):
                cv2.line(self.paint_window, self.last_point, base_index, (self.bg_color_val, self.bg_color_val, self.bg_color_val), 35)
            self.last_point = base_index
            return

        if is_index_up and is_middle_up:  
            if self.is_drawing: self.finalize_action()
            self.is_drawing = False
            self.last_point = (0,0)
            self.point_buffer.clear()

            self.check_slider_hover_events(index_tip, scale_factor)
            self.sample_wheel_color(index_tip, scale_factor)

            for button in self.buttons:
                if button.is_clicked(index_tip, scale_factor):
                    self.handle_button_click(button.key_id)

        elif is_index_up and not is_middle_up:  
            self.slider_visible = False  
            self.wheel_visible = False  
            
            if self.current_mode == "ERASER":
                if distance_pinch > int(55 * scale_factor):  
                    self.current_mode = self.pre_pinch_mode  
                    self.is_drawing = False
                    self.last_point = (0, 0)
                    self.point_buffer.clear()
                else:
                    return
            
            base_index = (int(index_tip[0] / scale_factor), int(index_tip[1] / scale_factor))
            if not self.is_drawing:
                self.is_drawing = True
                self.shape_start_point = base_index
                self.letter_points = []
                self.point_buffer.clear()
                self.last_point = base_index

            self.perform_action(base_index)
        else:
            if self.is_drawing: self.finalize_action()
            self.is_drawing = False
            self.last_point = (0,0)
            self.point_buffer.clear()

    def perform_action(self, base_finger_tip):
        self.point_buffer.append(base_finger_tip)
        draw_point = (sum(p[0] for p in self.point_buffer) // len(self.point_buffer), sum(p[1] for p in self.point_buffer) // len(self.point_buffer)) if len(self.point_buffer) > 1 else base_finger_tip

        if self.current_mode in ["DRAW", "ERASER"]:
            color = (self.bg_color_val, self.bg_color_val, self.bg_color_val) if self.current_mode == "ERASER" else self.current_color
            brush_w = 35 if self.current_mode == "ERASER" else self.thickness
            if self.last_point != (0,0):
                cv2.line(self.paint_window, self.last_point, draw_point, color, brush_w)
            if self.auto_write_mode and self.current_mode == "DRAW":
                self.letter_points.append(draw_point)
            self.last_point = draw_point
        elif self.current_mode in ["RECT", "CIRCLE", "CUBE"]:
            self.last_point = draw_point

    def finalize_action(self):
        raw_last_point = self.point_buffer[-1] if self.point_buffer else self.last_point
        if self.shape_start_point is None or raw_last_point == (0,0): return
        
        if self.current_mode == "RECT":
            cv2.rectangle(self.paint_window, self.shape_start_point, raw_last_point, self.current_color, self.thickness)
            self.save_canvas_snapshot()
        elif self.current_mode == "CIRCLE":
            radius = int(np.linalg.norm(np.array(self.shape_start_point) - np.array(raw_last_point)))
            cv2.circle(self.paint_window, self.shape_start_point, radius, self.current_color, self.thickness)
            self.save_canvas_snapshot()
        elif self.current_mode == "CUBE":
            self.draw_cube(self.shape_start_point, raw_last_point, self.paint_window)
            self.save_canvas_snapshot()
        elif self.current_mode == "FILL":
            cv2.floodFill(self.paint_window, None, seedPoint=raw_last_point, newVal=self.current_color)
            self.save_canvas_snapshot()
        elif self.current_mode in ["DRAW", "ERASER"]:
            if self.auto_write_mode and self.current_mode == "DRAW" and len(self.letter_points) > 5:
                self.perform_ocr()
            else:
                self.save_canvas_snapshot()
                self.letter_points.clear()

    def handle_button_click(self, name):
        if name in ["RECT", "CIRCLE", "CUBE", "FILL", "DRAW"]:
            self.current_mode = name
            self.pre_pinch_mode = name  
            if name == "DRAW" and self.current_color == (self.bg_color_val, self.bg_color_val, self.bg_color_val): 
                self.current_color = (0,0,0) if not self.dark_mode_active else (255,255,255)
        elif name == "ERASER":
            self.current_mode = "ERASER"
        elif name == "SIZE_P":
            self.slider_visible = not self.slider_visible
            self.wheel_visible = False
        elif name == "WHEEL_T":
            self.wheel_visible = not self.wheel_visible
            self.slider_visible = False
        elif name == "AUTO-W":
            self.auto_write_mode = not self.auto_write_mode
            self.letter_points.clear()
        elif name == "THEME":
            self.toggle_theme_workspace()
        elif name == "UNDO":
            self.trigger_undo()
        elif name == "REDO":
            self.trigger_redo()
        elif name == "CLEAR":
            self.paint_window.fill(self.bg_color_val)
            self.generate_grid_notebook()
            self.save_canvas_snapshot()
            self._reset_text_cursor()
            self.letter_points.clear()
        elif name == "SAVE":
            cv2.imwrite(f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", self.paint_window)
        elif name in [c[0] for c in self.colors_list]:
            for btn in self.buttons:
                if btn.key_id == name:
                    self.current_color = btn.color
            self.current_mode = "DRAW"
            self.pre_pinch_mode = "DRAW"

    def draw_all_ui(self, image, scale_factor, text_color_lbl):
        for button in self.buttons:
            is_active = (button.key_id == self.current_mode) or (button.key_id == "AUTO-W" and self.auto_write_mode) or \
                        (button.key_id == "SIZE_P" and self.slider_visible) or (button.key_id == "WHEEL_T" and self.wheel_visible) or \
                        (button.is_color_preset and button.color == self.current_color and self.current_mode not in ["ERASER"])
            image = button.draw(image, scale_factor, highlight=is_active, hovered=False, text_color=text_color_lbl)
        return image

    def perform_ocr(self):
        if not self.letter_points: return
        min_x = max(0, min(p[0] for p in self.letter_points) - 20)
        max_x = min(self.paint_window.shape[1], max(p[0] for p in self.letter_points) + 20)
        min_y = max(self.header_height + 5, min(p[1] for p in self.letter_points) - 20)
        max_y = min(self.paint_window.shape[0], max(p[1] for p in self.letter_points) + 20)
        self.letter_points.clear()

        if min_x >= max_x or min_y >= max_y: return
        letter_roi = self.paint_window[min_y:max_y, min_x:max_x].copy()
        
        cv2.rectangle(self.paint_window, (min_x, min_y), (max_x, max_y), (self.bg_color_val, self.bg_color_val, self.bg_color_val), -1)
        self.generate_grid_notebook()

        if letter_roi.size > 0:
            char_idx = self.recognize_character(letter_roi)
            if char_idx is not None:
                recognized_char = self.label_mapping[char_idx]
                
                pil_img = Image.fromarray(self.paint_window)
                draw = ImageDraw.Draw(pil_img)
                
                try: text_width = self.font.getbbox(recognized_char)[2] - self.font.getbbox(recognized_char)[0]
                except AttributeError: text_width = self.font.getsize(recognized_char)[0]

                if self.text_cursor_x + text_width > self.screen_width - self.text_margin:
                    self.text_cursor_x = self.text_margin
                    self.text_cursor_y += self.line_height

                draw.text((self.text_cursor_x, self.text_cursor_y), recognized_char, font=self.font, fill=self.ink_color_val)
                self.paint_window = np.array(pil_img)
                self.text_cursor_x += text_width + (FONT_SIZE // 4)
                self.save_canvas_snapshot()

    def recognize_character(self, img):
        if self.model is None: return None
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.dark_mode_active:
            _, img_thresh = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)
        else:
            _, img_thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
            
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        digit_roi = img_thresh[y:y+h, x:x+w]

        max_dim = max(w, h)
        if max_dim == 0: return None
        target_size = 20
        new_w = target_size if w > h else int(w * (target_size / h))
        new_h = target_size if h >= w else int(h * (target_size / w))
        
        if new_w <= 0 or new_h <= 0: return None
        digit_resized = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        processed_img = np.zeros((28, 28), dtype=np.uint8)
        processed_img[(28-new_h)//2:(28-new_h)//2+new_h, (28-new_w)//2:(28-new_w)//2+new_w] = digit_resized
        img_reshaped = processed_img.reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        prediction = self.model.predict(img_reshaped, verbose=0)
        confidence = np.max(prediction)
        
        print(f"AI Model Tracker: Classified '{self.label_mapping[np.argmax(prediction)]}' with confidence: {confidence:.2f}")
        if confidence > 0.15: return np.argmax(prediction)
        return None

    def draw_shape_preview(self, p1, p2, image):
        color = self.current_color
        if self.current_mode == "RECT": cv2.rectangle(image, p1, p2, color, 2)
        elif self.current_mode == "CIRCLE":
            radius = int(np.linalg.norm(np.array(p1) - np.array(p2)))
            cv2.circle(image, p1, radius, color, 2)
        elif self.current_mode == "CUBE": self.draw_cube(p1, p2, image)

    def draw_cube(self, p1, p2, image):
        color = self.current_color
        cv2.rectangle(image, p1, p2, color, 2)
        offset = int(np.linalg.norm(np.array(p1) - np.array(p2)) * 0.35)
        p3, p4 = (p1[0] + offset, p1[1] - offset), (p2[0] + offset, p2[1] - offset)
        cv2.rectangle(image, p3, p4, color, 2)
        cv2.line(image, p1, p3, color, 2); cv2.line(image, (p2[0], p1[1]), (p4[0], p3[1]), color, 2)
        cv2.line(image, (p1[0], p2[1]), (p3[0], p4[1]), color, 2); cv2.line(image, p2, p4, color, 2)

if __name__ == "__main__":
    app = AirCanvas()
    app.run()