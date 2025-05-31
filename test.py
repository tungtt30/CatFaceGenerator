import tkinter as tk
from tkinter import scrolledtext
import subprocess
import threading

class PythonShellApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Interactive Shell GUI")
        self.root.geometry("700x600")

        self.python_process = None

        # Label
        label = tk.Label(root, text="Giao diện Python Interactive Shell", font=("Arial", 12))
        label.pack(pady=5)

        # Output area
        self.output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=20, font=("Courier", 10))
        self.output_box.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Entry for user input
        self.command_entry = tk.Entry(root, font=("Arial", 12))
        self.command_entry.pack(pady=5, padx=10, fill=tk.X)
        self.command_entry.bind("<Return>", self.run_python_command)

        # Buttons for Python commands
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        # Predefined Python commands
        tk.Button(button_frame, text="In 'Hello, World!'", command=lambda: self.send_python_command("print('Hello, World!')")).grid(row=0, column=0, padx=10, pady=5)
        tk.Button(button_frame, text="Tính 2 + 2", command=lambda: self.send_python_command("print(2 + 2)")).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(button_frame, text="Vòng lặp 1 -> 5", command=lambda: self.send_python_command("for i in range(1, 6): print(i)")).grid(row=0, column=2, padx=10, pady=5)
        tk.Button(button_frame, text="Kiểm tra 5 > 3", command=lambda: self.send_python_command("print(5 > 3)")).grid(row=0, column=3, padx=10, pady=5)

        # Buttons to control the Python shell
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)

        start_button = tk.Button(control_frame, text="Start Python", command=self.start_python, bg="green", fg="white")
        stop_button = tk.Button(control_frame, text="Stop Python", command=self.stop_python, bg="red", fg="white")
        start_button.grid(row=0, column=0, padx=10)
        stop_button.grid(row=0, column=1, padx=10)

    def start_python(self):
        """
        Start the Python interactive shell process.
        """
        if self.python_process is None or self.python_process.poll() is not None:
            self.python_process = subprocess.Popen(
                ["python", "-u"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            threading.Thread(target=self.read_output, daemon=True).start()
            self.output_box.insert(tk.END, "Python interactive shell started.\n>>> ")
        else:
            self.output_box.insert(tk.END, "Python interactive shell is already running.\n>>> ")

    def stop_python(self):
        """
        Stop the Python interactive shell process.
        """
        if self.python_process:
            self.python_process.terminate()
            self.python_process = None
            self.output_box.insert(tk.END, "\nPython interactive shell stopped.\n")

    def run_python_command(self, event=None):
        """
        Send a command to the Python interactive shell.
        """
        command = self.command_entry.get()
        if command.strip() and self.python_process:
            self.send_python_command(command)
        elif not self.python_process:
            self.output_box.insert(tk.END, "Python interactive shell is not running.\n")
        self.command_entry.delete(0, tk.END)

    def send_python_command(self, command):
        """
        Helper function to send a command to the Python shell.
        """
        try:
            if self.python_process:
                self.python_process.stdin.write(command + "\n")
                self.python_process.stdin.flush()
                self.output_box.insert(tk.END, f">>> {command}\n")
            else:
                self.output_box.insert(tk.END, "Python interactive shell is not running.\n")
        except Exception as e:
            self.output_box.insert(tk.END, f"Error sending command: {e}\n")

    def read_output(self):
        """
        Continuously read output from the Python interactive shell.
        """
        while self.python_process and self.python_process.poll() is None:
            output = self.python_process.stdout.readline()
            if output:
                self.output_box.insert(tk.END, output)
                self.output_box.see(tk.END)
            error = self.python_process.stderr.readline()
            if error:
                self.output_box.insert(tk.END, f"Error: {error}\n")
                self.output_box.see(tk.END)

# Run the app
root = tk.Tk()
app = PythonShellApp(root)
root.mainloop()
