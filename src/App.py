from tkinter import Tk, Toplevel, Label, Button
from tkinter import filedialog
from utils.predict import predict


win = Tk()

win.geometry("800x450")


def open_file():
    filepath = filedialog.askopenfilename(title="Open an Audio File", filetypes=(
        ("audio", "*.mp3"), ("audio", "*.mp3")))
    result = predict(filepath)
    top = Toplevel(win)
    top.geometry("750x250")
    top.title("Result Window")
    Label(top, text=result, font=('Mistral 18 bold')).place(x=375, y=125)


button = Button(win, text="Predict", command=open_file)
button.place(x=400, y=225)
button.pack()

win.mainloop()
