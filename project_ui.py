import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
from tkinter import filedialog

try:
    import customtkinter as ctk
except ImportError as exc:
    raise ImportError("customtkinter is required. Install with: pip install customtkinter") from exc


class ProjectUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.title("PaperAPI Control Center")
        self.geometry("1024x720")
        self.minsize(900, 620)

        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.process = None
        self.log_queue = queue.Queue()

        self.script_var = ctk.StringVar(value="main.py")
        self.args_var = ctk.StringVar(value="")
        self.python_var = ctk.StringVar(value=sys.executable)
        self.stdin_var = ctk.StringVar(value="")
        self.status_var = ctk.StringVar(value="Idle")

        self._build_ui()
        self.after(100, self._drain_log_queue)

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        header = ctk.CTkFrame(self, corner_radius=24, fg_color="#0f172a")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 8))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Indoor Vision",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#7dd3fc",
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 2))

        ctk.CTkLabel(
            header,
            text="Run scripts, provide interactive stdin, and monitor output in one place.",
            font=ctk.CTkFont(size=13),
            text_color="#cbd5e1",
        ).grid(row=1, column=0, sticky="w", padx=16, pady=(0, 12))

        control = ctk.CTkFrame(self, corner_radius=24, fg_color="#111827")
        control.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        for i in range(4):
            control.grid_columnconfigure(i, weight=1 if i in (1, 3) else 0)

        ctk.CTkLabel(control, text="Script", text_color="#e2e8f0").grid(row=0, column=0, sticky="w", padx=(16, 8), pady=(14, 8))
        self.script_combo = ctk.CTkComboBox(
            control,
            variable=self.script_var,
            values=["main.py", "vista400k_ingest.py", "manage_chroma.py", "view_collection.py"],
            corner_radius=20,
            dropdown_fg_color="#1f2937",
            fg_color="#1f2937",
            border_color="#334155",
        )
        self.script_combo.grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=(14, 8))

        ctk.CTkLabel(control, text="Arguments", text_color="#e2e8f0").grid(row=0, column=2, sticky="w", padx=(0, 8), pady=(14, 8))
        self.args_entry = ctk.CTkEntry(
            control,
            textvariable=self.args_var,
            corner_radius=20,
            fg_color="#1f2937",
            border_color="#334155",
        )
        self.args_entry.grid(row=0, column=3, sticky="ew", padx=(0, 16), pady=(14, 8))

        ctk.CTkLabel(control, text="Python", text_color="#e2e8f0").grid(row=1, column=0, sticky="w", padx=(16, 8), pady=(0, 14))
        self.python_entry = ctk.CTkEntry(
            control,
            textvariable=self.python_var,
            corner_radius=20,
            fg_color="#1f2937",
            border_color="#334155",
        )
        self.python_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 12), pady=(0, 14))

        ctk.CTkButton(
            control,
            text="Browse",
            command=self._browse_python,
            corner_radius=20,
            fg_color="#334155",
            hover_color="#475569",
        ).grid(row=1, column=3, sticky="e", padx=(0, 16), pady=(0, 14))

        action = ctk.CTkFrame(self, corner_radius=24, fg_color="#111827")
        action.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))
        action.grid_columnconfigure(5, weight=1)

        self.run_btn = ctk.CTkButton(
            action,
            text="Run",
            command=self.run_script,
            corner_radius=20,
            fg_color="#22c55e",
            hover_color="#16a34a",
            text_color="#052e16",
            width=96,
        )
        self.run_btn.grid(row=0, column=0, padx=(16, 8), pady=12)

        self.stop_btn = ctk.CTkButton(
            action,
            text="Stop",
            command=self.stop_script,
            corner_radius=20,
            fg_color="#ef4444",
            hover_color="#dc2626",
            text_color="#fff1f2",
            width=96,
            state="disabled",
        )
        self.stop_btn.grid(row=0, column=1, padx=8, pady=12)

        ctk.CTkButton(
            action,
            text="Clear Log",
            command=self.clear_log,
            corner_radius=20,
            fg_color="#334155",
            hover_color="#475569",
            width=120,
        ).grid(row=0, column=2, padx=8, pady=12)

        ctk.CTkButton(
            action,
            text="Open Project Folder",
            command=self._open_project_folder,
            corner_radius=20,
            fg_color="#334155",
            hover_color="#475569",
            width=170,
        ).grid(row=0, column=3, padx=8, pady=12)

        ctk.CTkLabel(
            action,
            textvariable=self.status_var,
            text_color="#cbd5e1",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=6, padx=(8, 16), pady=12, sticky="e")

        stdin_frame = ctk.CTkFrame(self, corner_radius=24, fg_color="#111827")
        stdin_frame.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 8))
        stdin_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(stdin_frame, text="Stdin Input", text_color="#e2e8f0").grid(row=0, column=0, padx=(16, 8), pady=12, sticky="w")
        self.stdin_entry = ctk.CTkEntry(
            stdin_frame,
            textvariable=self.stdin_var,
            corner_radius=20,
            fg_color="#1f2937",
            border_color="#334155",
        )
        self.stdin_entry.grid(row=0, column=1, padx=(0, 8), pady=12, sticky="ew")
        self.stdin_entry.bind("<Return>", lambda _event: self.send_input())

        self.send_btn = ctk.CTkButton(
            stdin_frame,
            text="Send",
            command=self.send_input,
            corner_radius=20,
            fg_color="#334155",
            hover_color="#475569",
            width=100,
            state="disabled",
        )
        self.send_btn.grid(row=0, column=2, padx=(0, 16), pady=12)

        log_frame = ctk.CTkFrame(self, corner_radius=24, fg_color="#111827")
        log_frame.grid(row=4, column=0, sticky="nsew", padx=14, pady=(0, 14))
        log_frame.grid_rowconfigure(1, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(log_frame, text="Live Output", text_color="#e2e8f0").grid(row=0, column=0, sticky="w", padx=16, pady=(12, 6))
        self.log = ctk.CTkTextbox(
            log_frame,
            corner_radius=20,
            fg_color="#0b1220",
            text_color="#dbeafe",
            border_width=1,
            border_color="#223252",
            font=ctk.CTkFont(family="Consolas", size=12),
            wrap="word",
        )
        self.log.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 14))
        self._append_log(f"Working directory: {self.root_dir}\n")
        self._append_log("Ready.\n")

    def _append_log(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def _drain_log_queue(self):
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(line)
        self.after(100, self._drain_log_queue)

    def _browse_python(self):
        selected = filedialog.askopenfilename(
            title="Select Python Executable",
            filetypes=[("Python", "python.exe"), ("All Files", "*.*")],
        )
        if selected:
            self.python_var.set(selected)

    def _open_project_folder(self):
        if os.name == "nt":
            os.startfile(self.root_dir)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", self.root_dir])
        else:
            subprocess.Popen(["xdg-open", self.root_dir])

    def _set_running_state(self, running):
        self.run_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")
        self.send_btn.configure(state="normal" if running else "disabled")
        self.status_var.set("Running" if running else "Idle")

    def _read_stream(self, stream):
        try:
            for line in iter(stream.readline, ""):
                self.log_queue.put(line)
        finally:
            stream.close()

    def run_script(self):
        if self.process is not None:
            self._append_log("A process is already running.\n")
            return

        script = self.script_var.get().strip()
        python_exe = self.python_var.get().strip() or sys.executable
        raw_args = self.args_var.get().strip()

        if not script:
            self._append_log("Please provide a script name.\n")
            return

        script_path = os.path.join(self.root_dir, script)
        if not os.path.exists(script_path):
            self._append_log(f"Script not found: {script_path}\n")
            return

        cmd = [python_exe, "-u", script_path]
        if raw_args:
            cmd.extend(shlex.split(raw_args))

        self._append_log(f"\n$ {' '.join(cmd)}\n")
        self._set_running_state(True)

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self.process = subprocess.Popen(
            cmd,
            cwd=self.root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )

        threading.Thread(target=self._read_stream, args=(self.process.stdout,), daemon=True).start()
        threading.Thread(target=self._wait_for_process, daemon=True).start()

    def _wait_for_process(self):
        rc = self.process.wait()
        self.log_queue.put(f"\nProcess finished with exit code {rc}.\n")
        self.process = None
        self.after(0, lambda: self._set_running_state(False))

    def stop_script(self):
        if self.process is None:
            return

        try:
            if os.name == "nt":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.terminate()
        except Exception:
            self.process.terminate()
        self._append_log("Stop signal sent.\n")

    def clear_log(self):
        self.log.delete("1.0", "end")

    def send_input(self):
        if self.process is None or self.process.stdin is None:
            self._append_log("No running process to receive input.\n")
            return

        value = self.stdin_var.get()
        self.stdin_var.set("")
        try:
            self.process.stdin.write(value + "\n")
            self.process.stdin.flush()
            self._append_log(f">>> {value}\n")
        except Exception as exc:
            self._append_log(f"Failed to send input: {exc}\n")


if __name__ == "__main__":
    app = ProjectUI()
    app.mainloop()
