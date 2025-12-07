import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

from autodeepseek import AutoDeepSeek

class AutoDeepSeekGUI:
    def __init__(self, root):
        self.root = root
        root.title("AutoDeepSeek")
        self.agent = AutoDeepSeek()
        self.task_thread = None

        self.input_field = tk.Text(root, height=3)
        self.input_field.pack(fill=tk.X, padx=5, pady=5)

        button_frame = tk.Frame(root)
        button_frame.pack(fill=tk.X, padx=5)

        self.run_button = tk.Button(button_frame, text="Run", command=self.run_task)
        self.run_button.pack(side=tk.LEFT)

        self.abort_button = tk.Button(button_frame, text="Abort", state=tk.DISABLED, command=self.abort_task)
        self.abort_button.pack(side=tk.LEFT, padx=5)

        self.output_area = ScrolledText(root, height=20)
        self.output_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def run_task(self):
        task = self.input_field.get("1.0", tk.END).strip()
        if not task:
            return
        self.run_button.config(state=tk.DISABLED)
        self.abort_button.config(state=tk.NORMAL)
        self.output_area.insert(tk.END, f"> {task}\n")
        self.task_thread = threading.Thread(target=self.execute_task, args=(task,))
        self.task_thread.start()

    def execute_task(self, task):
        try:
            result = self.agent.complete_task(task)
            self.output_area.insert(tk.END, result + "\n")
        except Exception as e:
            self.output_area.insert(tk.END, f"Error: {e}\n")
        finally:
            self.abort_button.config(state=tk.DISABLED)
            self.run_button.config(state=tk.NORMAL)

    def abort_task(self):
        if self.agent:
            self.agent.request_abort()
        self.output_area.insert(tk.END, "Task aborted by user.\n")
        self.abort_button.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = AutoDeepSeekGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
