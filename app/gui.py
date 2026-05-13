import tkinter as tk
from tkinter import ttk
from threading import Thread
import queue
import time

from src.train import main


# =========================================================
# GLOBALS
# =========================================================

log_queue = queue.Queue()

spinner_running = False

start_time = None


# =========================================================
# LOGGER
# =========================================================

def gui_log(message):

    print(message)

    log_queue.put(str(message) + "\n")


# =========================================================
# TRAINING THREAD
# =========================================================

def run_training():

    global spinner_running
    global start_time

    spinner_running = True

    start_time = time.time()

    update_spinner()

    Thread(
        target=run_pipeline_safe,
        daemon=True
    ).start()


def run_pipeline_safe():

    global spinner_running

    try:

        gui_log("Starting Training Pipeline...")

        main()

        gui_log("Training Completed.")

    except Exception as error:

        gui_log(f"ERROR: {error}")

    finally:

        spinner_running = False


# =========================================================
# LOG UPDATE
# =========================================================

def update_logs():

    while not log_queue.empty():

        log_text.insert(
            tk.END,
            log_queue.get()
        )

        log_text.see(tk.END)

    root.after(200, update_logs)


# =========================================================
# SPINNER
# =========================================================

spinner_frames = [
    "Running.",
    "Running..",
    "Running..."
]

spinner_index = 0


def update_spinner():

    global spinner_index

    if spinner_running:

        spinner_label.config(
            text=spinner_frames[spinner_index]
        )

        spinner_index = (
            spinner_index + 1
        ) % len(spinner_frames)

        root.after(300, update_spinner)

    else:

        spinner_label.config(text="")


# =========================================================
# TIMER
# =========================================================

def update_timer():

    if spinner_running and start_time:

        elapsed = time.time() - start_time

        timer_label.config(
            text=f"Runtime: {elapsed:.2f} sec"
        )

    root.after(500, update_timer)


# =========================================================
# GUI WINDOW
# =========================================================

root = tk.Tk()

root.title(
    "Surface Roughness Prediction Dashboard"
)

root.state("zoomed")

root.minsize(900, 600)

# =========================================================
# LAYOUT
# =========================================================

main_frame = ttk.Frame(root)

main_frame.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)

# =========================================================
# LOG TERMINAL
# =========================================================

log_text = tk.Text(
    main_frame,
    bg="black",
    fg="lime",
    wrap="word",
    font=("Consolas", 10)
)

log_text.pack(
    fill="both",
    expand=True
)

# =========================================================
# BUTTON FRAME
# =========================================================

button_frame = ttk.Frame(root)

button_frame.pack(pady=10)

run_button = ttk.Button(
    button_frame,
    text="Run Training",
    command=run_training
)

run_button.grid(
    row=0,
    column=0,
    padx=10
)

# =========================================================
# STATUS LABELS
# =========================================================

spinner_label = ttk.Label(
    root,
    text="",
    font=("Consolas", 12)
)

spinner_label.pack()

timer_label = ttk.Label(
    root,
    text="Runtime: 0.00 sec",
    font=("Consolas", 12)
)

timer_label.pack()

# =========================================================
# LOOP
# =========================================================

update_logs()

update_timer()

root.mainloop()
