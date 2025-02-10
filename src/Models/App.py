import customtkinter as ctk
from tkinter import filedialog
from src.utils.predict import predict
from src.utils.preprocess import preprocess_data
from src.utils.train import train
import threading
import os
import queue
import time

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TK_SILENCE_DEPRECATION"] = "1"


class MusicGenreClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Music Genre Classifier")
        self.geometry("500x320")
        self.configure(bg="#1E1E1E")
        self._center_window()
        self._init_ui()
        self.after(200, self.focus_force)

        self.process_active = False
        self.start_time = None
        self.animations = {}
        self.preprocessing_complete = False

    def _center_window(self):
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 500) // 2
        y = (screen_height - 320) // 2
        self.geometry(f"+{x}+{y}")

    def _init_ui(self):
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(expand=True, fill="both")

        content_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        content_frame.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            content_frame,
            text="Music Genre Classifier",
            font=("SF Pro Display", 24, "bold"),
            text_color="white",
        ).pack(pady=(0, 15))
        self.status_label = ctk.CTkLabel(
            content_frame,
            text="Select an Audio File or Train",
            font=("SF Pro Display", 16, "bold"),
            text_color="white",
            wraplength=400,
        )
        self.status_label.pack(pady=5)

        self.dynamic_label = ctk.CTkLabel(
            content_frame,
            text="",
            font=("SF Pro Display", 16, "bold"),
            text_color="white",
        )
        self.dynamic_label.pack(pady=2)

        self.elapsed_label = ctk.CTkLabel(
            content_frame, text="", font=("SF Pro Display", 16), text_color="#8E8E93"
        )
        self.elapsed_label.pack(pady=2)

        self.progress_bar = ctk.CTkProgressBar(
            content_frame,
            mode="indeterminate",
            height=4,
            progress_color="#66C2FF",
            fg_color="#2A2A2A",
        )
        self.progress_bar.pack(fill="x", pady=15)

        button_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=(10, 0))

        button_style = {
            "font": ("SF Pro Display", 14),
            "corner_radius": 8,
            "height": 36,
            "width": 120,
            "fg_color": "#5A5A5A",
            "hover_color": "#48484A",
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
        self.process_active = True
        self.start_time = time.time()
        self.status_label.configure(text=label_text)
        self.progress_bar.start()
        self._update_elapsed_time()

    def _end_process(self, final_text):
        self.process_active = False
        self.progress_bar.stop()

        for label_prefix in list(self.animations.keys()):
            self._stop_animation(label_prefix)

        self.status_label.configure(text=final_text)
        self.dynamic_label.configure(text="Select an Audio File")

    def _update_elapsed_time(self):
        if self.process_active and self.start_time:
            elapsed_time = time.time() - self.start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            label = (
                f"Elapsed: {minutes:02}:{seconds:02}"
                if not self.preprocessing_complete
                else f"Training Elapsed: {minutes:02}:{seconds:02}"
            )
            self.elapsed_label.configure(text=label, text_color="#8E8E93")
            self.after(1000, self._update_elapsed_time)

    def _select_file_and_predict(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.mp3 *.wav"), ("All Files", "*.*")],
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
        result = predict(filepath, segment_queue=segment_queue).capitalize()
        self.after(0, self._animate_result, result)

    def _train(self):
        self._start_process("Training...")
        self.preprocessing_complete = False
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
        try:
            preprocess_data(label_queue=label_queue)
            self.preprocessing_complete = True
            self.start_time = time.time()

            accuracy = train(epoch_queue=epoch_queue)
            accuracy_queue.put(f"{accuracy:.2f}%")
        except Exception as e:
            self.after(0, lambda: self.status_label.configure(text=f"Error: {e}"))
        finally:
            self.after(0, lambda: self._end_process("Training complete!"))

    def _process_queue(self, queue_obj, label_prefix):
        if self.process_active:
            try:
                while not queue_obj.empty():
                    update_text = queue_obj.get_nowait()
                    if label_prefix == "Preprocessing":
                        if "complete" in update_text.lower():
                            self.preprocessing_complete = True
                            self._stop_animation(label_prefix)
                            self.dynamic_label.configure(
                                text=f"{label_prefix} {update_text}"
                            )
                        else:
                            self._start_animation(label_prefix, update_text)
                    elif label_prefix == "Epoch" and self.preprocessing_complete:
                        self._start_animation(label_prefix, update_text)
                    elif label_prefix == "Test Accuracy":
                        self.dynamic_label.configure(
                            text=f"{label_prefix} {update_text}"
                        )
            except queue.Empty:
                pass
            self.after(100, self._process_queue, queue_obj, label_prefix)

    def _start_animation(self, label_prefix, base_text):
        self._stop_animation(label_prefix)
        self.animations[label_prefix] = {"base": base_text, "dots": 0, "after_id": None}
        self._animate_dots(label_prefix)

    def _stop_animation(self, label_prefix):
        if label_prefix in self.animations:
            anim = self.animations[label_prefix]
            if anim["after_id"]:
                self.after_cancel(anim["after_id"])
            del self.animations[label_prefix]

    def _animate_dots(self, label_prefix):
        if label_prefix not in self.animations:
            return
        anim = self.animations[label_prefix]
        anim["dots"] = (anim["dots"] % 3) + 1
        dots = "." * anim["dots"]
        self.dynamic_label.configure(text=f"{label_prefix} {anim['base']}{dots}")
        anim["after_id"] = self.after(500, self._animate_dots, label_prefix)

    def _animate_result(self, result):
        self._end_process("")
        self._typewriter_effect(result, 0)

    def _typewriter_effect(self, text, index):
        if index < len(text):
            current_text = self.status_label.cget("text") + text[index]
            self.status_label.configure(text=current_text)
            self.after(50, self._typewriter_effect, text, index + 1)
        else:
            self.status_label.configure(text_color="white")


if __name__ == "__main__":
    app = MusicGenreClassifierApp()
    app.mainloop()
