import customtkinter as ctk
from tkinter import filedialog
from src.utils.predict import predict
import threading
import os

# Suppress macOS debug logs
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TK_SILENCE_DEPRECATION"] = "1"


class MusicGenreClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Music Genre Classifier")
        self.geometry("480x250")
        self.configure(bg="#1C1C1E")
        self._center_window()
        self._init_ui()
        self.after(200, self.focus_force)

    def _center_window(self):
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (480 // 2)
        y = (screen_height // 2) - (250 // 2)
        self.geometry(f"+{x}+{y}")

    def _init_ui(self):
        # Main frame
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            main_frame,
            text="Music Genre Classifier",
            font=("SF Pro Display", 22, "bold"),
            text_color="white",
        ).pack(pady=10)

        self.label = ctk.CTkLabel(
            main_frame,
            text="Select an Audio File",
            font=("SF Pro Display", 16),
            text_color="#A1A1A1",
        )
        self.label.pack(pady=10)

        self.loading_indicator = ctk.CTkProgressBar(
            main_frame,
            orientation="horizontal",
            mode="indeterminate",
            width=180,
            progress_color=("white", "#A1A1A1"),
            fg_color="#333333",
            corner_radius=10,
        )
        self.loading_indicator.pack(pady=10)
        self.loading_indicator.stop()

        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(pady=10)

        ctk.CTkButton(
            button_frame,
            text="Choose File",
            font=("SF Pro Display", 14),
            corner_radius=25,
            fg_color="white",
            text_color="black",
            hover_color="#D9D9D9",
            command=self._select_file_and_predict,
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="Exit",
            font=("SF Pro Display", 14),
            corner_radius=25,
            fg_color="white",
            text_color="black",
            hover_color="#D9D9D9",
            command=self.quit,
        ).pack(side="left", padx=5)

    def _select_file_and_predict(self):
        filepath = filedialog.askopenfilename(
            title="Open an Audio File",
            filetypes=(
                ("MP3 Files", "*.mp3"),
                ("WAV Files", "*.wav"),
                ("All Files", "*.*"),
            ),
            initialdir="~/Downloads",
        )
        if filepath:
            self.label.configure(text="Processing...")
            self.loading_indicator.start()
            threading.Thread(
                target=self._predict_genre, args=(filepath,), daemon=True
            ).start()

    def _predict_genre(self, filepath):
        try:
            result = predict(filepath).capitalize()
            self.after(0, self._animate_result, result)
        except Exception as e:
            self.after(0, lambda: self.label.configure(text=f"Error: {str(e)}"))
            self.loading_indicator.stop()

    def _animate_result(self, result):
        self.loading_indicator.stop()
        self.loading_indicator.configure(mode="determinate")

        def progress_animation(step=0):
            if step <= 100:
                self.loading_indicator.set(step / 100)
                self.after(5, progress_animation, step + 2)
            else:
                self.label.configure(text="")
                self._typewriter_effect(result, 0)

        progress_animation()

    def _typewriter_effect(self, text, index):
        if index < len(text):
            current_text = self.label.cget("text") + text[index]
            self.label.configure(text=current_text)
            self.after(50, self._typewriter_effect, text, index + 1)
