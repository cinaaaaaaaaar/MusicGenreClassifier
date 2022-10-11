from tkinter import Tk, Label, Button
from tkinter import filedialog
from utils.predict import predict
win = Tk()
win.geometry("500x300")

label = Label(win, text="Open a File", font=('Mistral 18 bold'))
label.grid()
label.pack()
filepath = filedialog.askopenfilename(title="Open an Audio File", filetypes=(
    ("audio", "*.mp3"), ("audio", "*.mp3")))
result = predict(filepath)
print(result)
label.config(text=result)
win.mainloop()
