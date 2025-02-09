import customtkinter as ctk
from tkinter import filedialog
from src.utils.predict import predict
from src.utils.preprocess import preprocess_data
from src.utils.train import train
import threading
import os
import queue
import time

# Suppress macOS debug logs
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TK_SILENCE_DEPRECATION"] = "1"


class MusicGenreClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Music Genre Classifier")
        self.geometry("500x320")
        self.configure(bg="#1C1C1E")
        self._center_window()
        self._init_ui()
        self.after(200, self.focus_force)

        # Shared state variables
        self.process_active = False
        self.start_time = None

    def _center_window(self):
        """Centers the window on the screen."""
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 500) // 2
        y = (screen_height - 320) // 2
        self.geometry(f"+{x}+{y}")

    def _init_ui(self):
        """Initializes the UI components."""
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(expand=True, fill="both")  # Ensure it takes full space

        content_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        content_frame.place(relx=0.5, rely=0.5, anchor="center")  # Center the content

        # Title
        ctk.CTkLabel(
            content_frame,
            text="Music Genre Classifier",
            font=("SF Pro Display", 24, "bold"),
            text_color="white",
        ).pack(pady=(0, 15))

        # Status label
        self.status_label = ctk.CTkLabel(
            content_frame,
            text="Select an Audio File or Train",
            font=("SF Pro Display", 16, "bold"),
            text_color="white",
            wraplength=400,
        )
        self.status_label.pack(pady=5)

        # Dynamic label (for segments, epochs, accuracy)
        self.dynamic_label = ctk.CTkLabel(
            content_frame,
            text="",
            font=("SF Pro Display", 16, "bold"),
            text_color="white",
        )
        self.dynamic_label.pack(pady=2)

        # Elapsed time label (remains visible after training)
        self.elapsed_label = ctk.CTkLabel(
            content_frame, text="", font=("SF Pro Display", 16), text_color="#8E8E93"
        )
        self.elapsed_label.pack(pady=2)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            content_frame,
            orientation="horizontal",
            mode="indeterminate",
            height=4,
            progress_color="#0A84FF",
            fg_color="#2C2C2E",
            corner_radius=2,
        )
        self.progress_bar.pack(fill="x", pady=15)

        # Buttons
        button_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=(10, 0))

        button_style = {
            "font": ("SF Pro Display", 14),
            "corner_radius": 8,
            "height": 36,
            "width": 120,
            "border_width": 0,
            "fg_color": "#363638",
            "hover_color": "#48484A",
            "text_color": "white",
        }

        ctk.CTkButton(
            button_frame,
            text="Choose File",
            command=self._select_file_and_predict,
            **button_style,
        ).pack(side="left", padx=4, expand=True)
        ctk.CTkButton(
            button_frame, text="Train", command=self._train, **button_style
        ).pack(side="left", padx=4, expand=True)
        ctk.CTkButton(
            button_frame, text="Exit", command=self.quit, **button_style
        ).pack(side="left", padx=4, expand=True)

    def _start_process(self, label_text, keep_progress_blue=True):
        """Starts a process with a progress bar and timer."""
        self.process_active = True
        self.start_time = time.time()
        self.status_label.configure(text=label_text)
        self.progress_bar.configure(
            progress_color="#0A84FF" if keep_progress_blue else "#30D158"
        )
        self.progress_bar.start()
        self._update_elapsed_time()

    def _end_process(self, final_text):
        """Ends a process and updates UI elements."""
        self.process_active = False
        self.progress_bar.stop()
        self.status_label.configure(text=final_text)

    def _update_elapsed_time(self):
        """Updates elapsed time during a process."""
        if self.process_active and self.start_time:
            elapsed_time = time.time() - self.start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            self.elapsed_label.configure(
                text=f"Elapsed: {minutes:02}:{seconds:02}", text_color="#8E8E93"
            )
            self.after(1000, self._update_elapsed_time)

    def _select_file_and_predict(self):
        """Handles file selection and starts prediction."""
        filepath = filedialog.askopenfilename(
            title="Open an Audio File",
            filetypes=[
                ("MP3 Files", "*.mp3"),
                ("WAV Files", "*.wav"),
                ("All Files", "*.*"),
            ],
            initialdir="~/Downloads",
        )
        if filepath:
            self._start_process("Processing...", keep_progress_blue=False)
            segment_queue = queue.Queue()
            threading.Thread(
                target=self._predict_genre, args=(filepath, segment_queue), daemon=True
            ).start()
            self.after(100, self._process_queue, segment_queue, "Segment")

    def _predict_genre(self, filepath, segment_queue):
        """Runs prediction and updates the queue."""
        result = predict(filepath, segment_queue=segment_queue).capitalize()
        self.after(0, self._animate_result, result)

    def _train(self):
        """Handles training initiation."""
        self._start_process("Training...")
        label_queue = queue.Queue()
        epoch_queue = queue.Queue()
        accuracy_queue = queue.Queue()
        threading.Thread(
            target=self._train_model,
            args=(label_queue, epoch_queue, accuracy_queue),
            daemon=True,
        ).start()
        self.after(100, self._process_queue, label_queue, "Preprocessing")
        self.after(100, self._process_queue, epoch_queue, "Epoch")
        self.after(100, self._process_queue, accuracy_queue, "Test Accuracy")

    def _train_model(self, label_queue, epoch_queue, accuracy_queue):
        """Runs the training process and gets model accuracy."""
        try:
            preprocess_data(label_queue=label_queue)
            accuracy = train(epoch_queue=epoch_queue)  # Returns test accuracy
            accuracy_queue.put(f"{accuracy:.2f}%")
        except Exception as e:
            self.after(0, lambda: self.status_label.configure(text=f"Error: {e}"))
        finally:
            self.after(0, lambda: self._end_process("Training complete."))

    def _process_queue(self, queue_obj, label_prefix):
        """Processes a queue and updates the UI dynamically."""
        if self.process_active:
            try:
                while not queue_obj.empty():
                    update_text = queue_obj.get_nowait()
                    self.dynamic_label.configure(text=f"{label_prefix}: {update_text}")
            except queue.Empty:
                pass
            self.after(100, self._process_queue, queue_obj, label_prefix)

    def _animate_result(self, result):
        """Handles the final result display."""
        self._end_process("")
        self._typewriter_effect(result, 0)

    def _typewriter_effect(self, text, index):
        """Displays text with a typewriter effect."""
        if index < len(text):
            current_text = self.status_label.cget("text") + text[index]
            self.status_label.configure(text=current_text, text_color="#30D158")
            self.after(50, self._typewriter_effect, text, index + 1)
        else:
            self.status_label.configure(text_color="white")


if __name__ == "__main__":
    app = MusicGenreClassifierApp()
    app.mainloop()
