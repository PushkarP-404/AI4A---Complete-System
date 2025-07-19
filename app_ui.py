import tkinter as tk
import subprocess

def run_script_1():
    label = text_input.get()
    number = number_input.get()
    subprocess.Popen(["python", "main.py", label, number])

def run_script_2():
    label = text_input.get()
    number = number_input.get()
    subprocess.Popen(["python", "collect_data.py", label, number])

def run_script_3():
    subprocess.Popen(["python", "train_model.py"])

# Main window
root = tk.Tk()
root.title("AI Access App")
root.configure(bg="#1e1e1e")
root.geometry("1200x400")

# Frame for buttons
frame = tk.Frame(root, bg="#1e1e1e")
frame.pack(pady=30)

# Button style
btn_style = {
    "bg": "#292929",
    "fg": "#ffffff",
    "activebackground": "#555555",
    "activeforeground": "#ffffff",
    "width": 25,
    "height": 3,
    "font": ("Segoe UI", 12, "bold"),
    "relief": "raised",
    "bd": 3
}

# Button + Description
def add_button_with_label(text, command, desc):
    sub_frame = tk.Frame(frame, bg="#1e1e1e")
    btn = tk.Button(sub_frame, text=text, command=command, **btn_style)
    btn.pack()
    label = tk.Label(sub_frame, text=desc, fg="#cccccc", bg="#1e1e1e", font=("Segoe UI", 10))
    label.pack(pady=(10, 0))
    sub_frame.pack(side=tk.LEFT, padx=25)

add_button_with_label("Run main AI App", run_script_1, "Starts the main real-time AI sign recognition")
add_button_with_label("Collect Data", run_script_2, "Records new signs into the dataset - Enter label/sign name and recording size")
add_button_with_label("Train New Model", run_script_3, "Trains new AI model using collected data")

# Input fields
input_frame = tk.Frame(root, bg="#1e1e1e")
input_frame.pack(pady=20)

tk.Label(input_frame, text="Sign Label:", fg="white", bg="#1e1e1e", font=("Segoe UI", 12)).grid(row=0, column=0, padx=10, pady=5, sticky='e')
text_input = tk.Entry(input_frame, font=("Segoe UI", 12), width=30)
text_input.grid(row=0, column=1, padx=10, pady=5)

tk.Label(input_frame, text="Sequence Count:", fg="white", bg="#1e1e1e", font=("Segoe UI", 12)).grid(row=1, column=0, padx=10, pady=5, sticky='e')
number_input = tk.Entry(input_frame, font=("Segoe UI", 12), width=10)
number_input.grid(row=1, column=1, padx=10, pady=5, sticky='w')

root.mainloop()
