import random
import math
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Elo update functions ---
def update_elo(R, opp_R, K_func, result, Rmin):
    E = 1.0 / (1 + 10 ** ((opp_R - R) / 400.0))
    K = K_func(R, Rmin)
    return max(R + K * (result - E), Rmin), K

def K_linear(R, Rmin, Kmax=40, c=0.5):
    return min(Kmax, c * max(R - Rmin, 0))

def K_sigmoid(R, Rmin, Kmax=40, tau=5):
    val = Kmax / (1 + math.exp(-(R - Rmin) / tau))
    return min(val, R - Rmin)

def K_power(R, Rmin, Kmax=40, alpha=0.5, p=0.5):
    val = alpha * ((R - Rmin) ** p)
    return min(val, Kmax, R - Rmin)

# --- Simulation ---
def simulate(mode, games, Kmax, c, tau, alpha, p, Rmin, win_prob):
    history_A, history_B, history_C = [1500], [1500], [1500]
    R_A, R_B, R_C = 1500, 1500, 1500
    Kmax_flag_A, Kmax_flag_B, Kmax_flag_C = [True], [True], [True]

    for i in range(games):
        opp_A = random.uniform(0.8*R_A, 1.2*R_A)
        opp_B = random.uniform(0.8*R_B, 1.2*R_B)
        opp_C = random.uniform(0.8*R_C, 1.2*R_C)

        if mode == "patternA":
            res = [0,0,1][i%3]
        elif mode == "patternB":
            res = 0
        elif mode == "patternC":
            res = 1 if random.random() < win_prob else 0

        # Linear
        K_val = K_linear(R_A, Rmin, Kmax, c)
        Kmax_flag_A.append(K_val == Kmax)
        R_A, _ = update_elo(R_A, opp_A, lambda R,Rmin: K_linear(R,Rmin,Kmax,c), res, Rmin)

        # Sigmoid
        K_val = K_sigmoid(R_B, Rmin, Kmax, tau)
        Kmax_flag_B.append(K_val == Kmax)
        R_B, _ = update_elo(R_B, opp_B, lambda R,Rmin: K_sigmoid(R,Rmin,Kmax,tau), res, Rmin)

        # Power
        K_val = K_power(R_C, Rmin, Kmax, alpha, p)
        Kmax_flag_C.append(K_val == Kmax)
        R_C, _ = update_elo(R_C, opp_C, lambda R,Rmin: K_power(R,Rmin,Kmax,alpha,p), res, Rmin)

        history_A.append(R_A)
        history_B.append(R_B)
        history_C.append(R_C)

    return history_A, history_B, history_C, Kmax_flag_A, Kmax_flag_B, Kmax_flag_C

# --- GUI Application ---
class EloApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Elo Simulation")
        self.root.configure(bg="#f9f9f9")

        self.mode = tk.StringVar(value="patternA")

        # Variables
        self.Kmax = tk.DoubleVar(value=40)
        self.Rmin = tk.DoubleVar(value=100)
        self.c = tk.DoubleVar(value=0.5)
        self.tau = tk.DoubleVar(value=5)
        self.alpha = tk.DoubleVar(value=0.5)
        self.p = tk.DoubleVar(value=0.5)
        self.win_prob = tk.DoubleVar(value=0.3)

        self.create_control_panel()
        self.create_plot_area()
        self.plot()

    def create_control_panel(self):
        panel = tk.Toplevel(self.root)
        panel.title("Controls")
        panel.configure(bg="#f0f0f0")
        panel.geometry("650x650")

        style = ttk.Style()
        style.configure("Blue.TButton", font=("Arial", 12, "bold"), foreground="black",
                        background="#4a90e2", wraplength=80)
        style.map("Blue.TButton",
                  foreground=[('active','black')],
                  background=[('active','#357ABD')])

        # Buttons row
        ttk.Button(panel, text="Lose\nLose\nWin", style="Blue.TButton", command=lambda: self.set_mode("patternA")).grid(row=0,column=0,sticky="nsew", padx=5, pady=5)
        ttk.Button(panel, text="Always\nLose", style="Blue.TButton", command=lambda: self.set_mode("patternB")).grid(row=0,column=1,sticky="nsew", padx=5, pady=5)
        ttk.Button(panel, text="Probabilistic", style="Blue.TButton", command=lambda: self.set_mode("patternC")).grid(row=0,column=2,sticky="nsew", padx=5, pady=5)

        panel.columnconfigure(0, weight=1)
        panel.columnconfigure(1, weight=1)
        panel.columnconfigure(2, weight=1)
        panel.rowconfigure(0, weight=1)

        # Formula frames
        formula_frames = [
            ("Linear K: K = min(Kmax, c*(R-Rmin))", [("c", self.c, 0.01, 1.0)]),
            ("Sigmoid K: K = min(Kmax/(1+exp(-(R-Rmin)/tau)), R-Rmin)", [("tau", self.tau, 0.1, 15)]),
            ("Power K: K = min(alpha*(R-Rmin)^p, Kmax, R-Rmin)", [("alpha", self.alpha, 0.01, 2.0), ("p (exponent)", self.p, 0.01, 2.0)]),
            ("Common variables", [("Kmax", self.Kmax, 1, 100), ("Rmin", self.Rmin, 0, 500), ("win_prob (0~1)", self.win_prob, 0.0, 1.0)])
        ]

        for i, (title, vars_list) in enumerate(formula_frames, start=1):
            frame = tk.LabelFrame(panel, text=title, font=("Arial",12,"bold"), bg="#f0f0f0")
            frame.grid(row=i, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
            panel.rowconfigure(i, weight=1)
            frame.columnconfigure(0, weight=1)
            frame.columnconfigure(1, weight=1)
            frame.columnconfigure(2, weight=3)
            for j, (name, var, vmin, vmax) in enumerate(vars_list):
                tk.Label(frame, text=name, font=("Arial",12), width=12, anchor="w", bg="#f0f0f0").grid(row=j,column=0, sticky="w", padx=5, pady=2)
                tk.Entry(frame, textvariable=var, width=6, font=("Arial",12)).grid(row=j,column=1, sticky="w", padx=5, pady=2)
                tk.Scale(frame, variable=var, from_=vmin, to=vmax, resolution=0.01, orient="horizontal",
                         command=lambda e: self.plot(), font=("Arial",10)).grid(row=j,column=2, sticky="nsew", padx=5, pady=2)

        for col in range(3):
            panel.columnconfigure(col, weight=1)
        for row in range(5):
            panel.rowconfigure(row, weight=1)

    def create_plot_area(self):
        self.fig, self.ax = plt.subplots(figsize=(8,5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def set_mode(self, mode):
        self.mode.set(mode)
        self.plot()

    def plot(self):
        hA, hB, hC, KflagA, KflagB, KflagC = simulate(
            self.mode.get(), 400,
            self.Kmax.get(), self.c.get(), self.tau.get(),
            self.alpha.get(), self.p.get(), self.Rmin.get(), self.win_prob.get()
        )
        self.ax.clear()
        indices = np.arange(len(hA))

        def plot_with_flags(history, Kflag, color, label):
            history = np.array(history)
            Kflag = np.array(Kflag)
            change_points = np.where(np.diff(Kflag.astype(int)) != 0)[0] + 1
            start = 0
            for cp in list(change_points) + [len(history)]:
                segment_x = indices[start:cp]
                segment_y = history[start:cp]
                lw = 3 if Kflag[start] else 1
                self.ax.plot(segment_x, segment_y, color=color, linewidth=lw)
                start = cp
            # 过渡细线段
            for cp in change_points:
                self.ax.plot(indices[cp-1:cp+1], history[cp-1:cp+1], color=color, linewidth=1)
            self.ax.plot([], [], color=color, linewidth=3, label=f"{label} (K=Kmax thick)")

        plot_with_flags(hA, KflagA, "#e74c3c", "K = linear")
        plot_with_flags(hB, KflagB, "#3498db", "K = sigmoid")
        plot_with_flags(hC, KflagC, "#2ecc71", "K = power law")

        self.ax.axhline(y=self.Rmin.get(), color="black", linestyle="--", label="Floor")
        self.ax.set_xlabel("Games", fontsize=12)
        self.ax.set_ylabel("Elo Rating", fontsize=12)
        self.ax.set_title("Elo Rating Evolution", fontsize=14)
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

# --- Main ---
if __name__ == "__main__":
    root = tk.Tk()
    app = EloApp(root)
    root.mainloop()
