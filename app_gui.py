import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import time

class SemanticResumeMatcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Semantic Resume Matcher")
        self.root.geometry("600x400")

        self.job_description_path = None
        self.resume_paths = []

        self.create_widgets()

    def create_widgets(self):
        # Buttons Frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Job Description Text Area
        jd_frame = tk.LabelFrame(self.root, text="Job Description (Paste or Type)")
        jd_frame.pack(pady=10, padx=10, fill=tk.X)

        self.jd_text_area = tk.Text(jd_frame, height=8, wrap=tk.WORD)
        self.jd_text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        jd_scrollbar = tk.Scrollbar(jd_frame, command=self.jd_text_area.yview)
        jd_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.jd_text_area.config(yscrollcommand=jd_scrollbar.set)

        # Buttons Frame (moved below JD text area for better layout)
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        self.upload_resumes_button = tk.Button(button_frame, text="Upload Resumes", command=self.upload_resumes)
        self.upload_resumes_button.pack(side=tk.LEFT, padx=10)

        self.find_candidates_button = tk.Button(button_frame, text="Find Candidates", command=self.find_candidates, state=tk.DISABLED)
        self.find_candidates_button.pack(side=tk.LEFT, padx=10)

        # Status Label
        self.status_label = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, padx=10, pady=5)

        # Create a Notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Create Frames for each tab
        self.results_tab = ttk.Frame(self.notebook)
        self.logs_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.results_tab, text="Results")
        self.notebook.add(self.logs_tab, text="Logs")

        # Results Listbox (moved to results_tab)
        self.results_listbox = tk.Listbox(self.results_tab)
        self.results_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(self.results_tab, orient="vertical", command=self.results_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")
        self.results_listbox.config(yscrollcommand=scrollbar.set)
        self.results_listbox.bind("<Control-c>", self.copy_selected_result)
        self.results_listbox.bind("<Control-C>", self.copy_selected_result) # For consistency, though usually one is enough

        # Logs Text Area (in logs_tab)
        self.log_text_area = tk.Text(self.logs_tab, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        log_scrollbar = tk.Scrollbar(self.logs_tab, command=self.log_text_area.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text_area.config(yscrollcommand=log_scrollbar.set)

    

    def upload_resumes(self):
        # Allow selecting multiple files or a directory
        # For simplicity, let's start with multiple file selection
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Document files", "*.pdf *.docx")]
        )
        if file_paths:
            self.resume_paths = list(file_paths)
            self.status_label.config(text=f"Uploaded {len(self.resume_paths)} resumes.")
            self.check_find_candidates_button_state()

    def check_find_candidates_button_state(self):
        jd_content = self.jd_text_area.get("1.0", tk.END).strip()
        if jd_content and self.resume_paths:
            self.find_candidates_button.config(state=tk.NORMAL)
        else:
            self.find_candidates_button.config(state=tk.DISABLED)

    def find_candidates(self):
        jd_content = self.jd_text_area.get("1.0", tk.END).strip()
        if not jd_content:
            messagebox.showwarning("Input Error", "Please enter or paste a job description.")
            self.status_label.config(text="Ready")
            return

        self.status_label.config(text="Processing... please wait.")
        self.results_listbox.delete(0, tk.END) # Clear previous results
        self.root.update_idletasks() # Update GUI to show processing message

        start_time = time.time() # Record start time

        from matching_engine import MatchingEngine
        engine = MatchingEngine()
        results, logs = engine.match_resumes_to_job_description_text(jd_content, self.resume_paths)

        end_time = time.time() # Record end time
        duration = end_time - start_time
        
        # Clear and display logs
        self.log_text_area.config(state=tk.NORMAL) # Enable editing
        self.log_text_area.delete("1.0", tk.END)
        if logs:
            self.log_text_area.insert(tk.END, "\n".join(logs))
        else:
            self.log_text_area.insert(tk.END, "No detailed logs available.")
        self.log_text_area.insert(tk.END, f"\n--- Processing Time: {duration:.2f} seconds ---") # Add duration to logs
        self.log_text_area.config(state=tk.DISABLED) # Disable editing

        if isinstance(results, str): # Check if an error message was returned
            self.status_label.config(text=results)
            messagebox.showerror("Error", results)
        elif results:
            self.status_label.config(text=f"Matching complete. Displaying results. Took {duration:.2f} seconds.") # Update status with duration
            for i, result in enumerate(results):
                self.results_listbox.insert(tk.END, f"{i+1}. {result['filename']} (Match Score: {result['score']:.2f})")
        else:
            self.status_label.config(text=f"No matching candidates found or an error occurred. Took {duration:.2f} seconds.") # Update status with duration
            messagebox.showinfo("No Results", "No matching candidates found or an error occurred during processing.")

        self.find_candidates_button.config(state=tk.NORMAL) # Re-enable button after processing

    def copy_selected_result(self, event=None):
        selected_indices = self.results_listbox.curselection()
        if not selected_indices:
            return

        selected_text = []
        for index in selected_indices:
            selected_text.append(self.results_listbox.get(index))

        if selected_text:
            self.root.clipboard_clear()
            self.root.clipboard_append("\n".join(selected_text))
            messagebox.showinfo("Copy", "Selected item(s) copied to clipboard.")


if __name__ == "__main__":
    root = tk.Tk()
    app = SemanticResumeMatcherApp(root)
    root.mainloop()
