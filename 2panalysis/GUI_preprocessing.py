import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import os
'''GUI for preprocessing 2P datasets'''

################################
def submit():
    global motion
    motion = v.get()
    global roi
    roi=v1.get()
    root.destroy()
    return motion, roi
def select_dataset():
    global dataset_path
    dataset_path=filedialog.askdirectory(initialdir='C:/',title='Please select Dataset-Folder')
    btn1.config(text=dataset_path)
def select_logbook():
    global log_path
    file=filedialog.askopenfile(initialdir='C:/',title='Please select Metadata-file', filetypes=[('Excel-Sheet', '.xlsx')]) 
    if file:
        log_path = file.name
        btn2.config(text=log_path)
################################
root = tk.Tk()
root.option_add("*Font", "Times 15")
root.title("2Photon Preprocessing")
frame = Frame(root, width=350, height=300)
frame.pack()
# Create an object of tkinter ImageTk
width = 200
height = 200
img = ImageTk.PhotoImage(Image.open(f"{os.getcwd()}/wiki/logo.png").resize((width,height), resample=Image.BICUBIC))
# Create a Label Widget to display the text or Image
label = Label(frame, image = img)
label.grid(column=1, row=0, pady=20)
label1 = tk.Label(frame, text='2Photon Imaging', font=("Times", 20, "bold"))
label1.grid(row=1, column=1, pady=5)
label2 = tk.Label(frame, text='-Batch Preprocessing-', font=("Times", 20, "bold"))
label2.grid(row=2, column=1, pady=(0,20))

parameter_frame = LabelFrame(frame, text='Please Select Processing Parameters')
parameter_frame.grid(row=3, column=1, padx=25, pady=5)

label3 = tk.Label(parameter_frame, text="Dataset-Path")
label3.grid(row=4, column=0, pady=(20,0))

btn1 = Button(parameter_frame, text='Click to select Path to Dataset-Folder', command=select_dataset)
btn1.grid(row=4, column=1, pady=(20,0))

label4 = tk.Label(parameter_frame, text="Metadata-Path")
label4.grid(row=5, column=0)
btn2 = Button(parameter_frame, text='Click to select Path to Metadata-file', command=select_logbook)
btn2.grid(row=5, column=1)

v = IntVar()
label5 = tk.Label(parameter_frame, text="Motion Correction?")
label5.grid(row=6, column =0, padx=50)
rb1 = Radiobutton(parameter_frame, text="Yes", variable=v, value=0)
rb1.grid(row=6, column =1)
rb2 = Radiobutton(parameter_frame, text="No", variable=v, value=1)
rb2.grid(row=6, column =2, padx=(0,50))

v1 = IntVar()
label6 = tk.Label(parameter_frame, text="Selecting ROIs?")
label6.grid(row=7, column =0)
rb1 = Radiobutton(parameter_frame, text="Yes", variable=v1, value=0)
rb1.grid(row=7, column =1)
rb2 = Radiobutton(parameter_frame, text="No", variable=v1, value=1)
rb2.grid(row=7, column =2, padx=(0,50))

submit_button = tk.Button(frame, text="Submit", command=submit)
submit_button.grid(row=8, column=1, pady=20)

root.mainloop()