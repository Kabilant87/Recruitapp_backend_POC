# RecruitSmart Application Architecture and Workflow

This document outlines the architecture of the RecruitSmart application and provides detailed workflow notes for its key functions.

## 1. High-Level Architecture

The RecruitSmart application consists of a graphical user interface (GUI) built with Tkinter (`app_gui.py`) and a backend matching engine (`matching_engine.py`) responsible for semantic resume matching.

```mermaid
graph TD
    A[User Interaction (GUI)] --> B(app_gui.py);
    B -- Job Description Text, Resume Files --> C(matching_engine.py);
    C -- Match Results, Detailed Logs --> B;
    B -- Display Results & Logs --> D[GUI Output];
    C -- Text Extraction (PDF/DOCX) --> E[File System];
    C -- Embedding Generation --> F[SentenceTransformer Model];
    C -- Similarity Search --> G[FAISS Index];
```

## 2. Workflow Notes for Functions

### `app_gui.py`

#### `SemanticResumeMatcherApp.__init__(self, root)`

*   **Purpose:** Initializes the main application window and sets up initial state variables.
*   **Inputs:**
    *   `root`: The Tkinter root window.
*   **Outputs:** None. Initializes instance variables.
*   **Workflow:**
    1.  Sets the root window, title, and initial geometry.
    2.  Initializes `self.job_description_path` and `self.resume_paths`.
    3.  Calls `self.create_widgets()` to build the GUI.

#### `SemanticResumeMatcherApp.create_widgets(self)`

*   **Purpose:** Creates and arranges all the GUI elements (buttons, text areas, listbox, tabs).
*   **Inputs:** `self` (instance of `SemanticResumeMatcherApp`).
*   **Outputs:** None. Populates the Tkinter window with widgets.
*   **Workflow:**
    1.  Creates frames for buttons and job description input.
    2.  Initializes `self.jd_text_area` for job description input with a scrollbar.
    3.  Creates "Upload Resumes" and "Find Candidates" buttons.
    4.  Initializes `self.status_label` to display application status.
    5.  Creates a `ttk.Notebook` (tabbed interface) with "Results" and "Logs" tabs.
    6.  Initializes `self.results_listbox` with a scrollbar within the "Results" tab.
    7.  Binds `Ctrl+C` to `self.copy_selected_result` for the results listbox.
    8.  Initializes `self.log_text_area` with a scrollbar within the "Logs" tab, set to read-only.

#### `SemanticResumeMatcherApp.upload_resumes(self)`

*   **Purpose:** Handles the file selection dialog for uploading resume files.
*   **Inputs:** `self`.
*   **Outputs:** None. Updates `self.resume_paths` and `self.status_label`.
*   **Workflow:**
    1.  Opens a file dialog allowing selection of multiple PDF or DOCX files.
    2.  If files are selected, updates `self.resume_paths` with the selected file paths.
    3.  Updates `self.status_label` to show the number of uploaded resumes.
    4.  Calls `self.check_find_candidates_button_state()` to enable/disable the "Find Candidates" button.

#### `SemanticResumeMatcherApp.check_find_candidates_button_state(self)`

*   **Purpose:** Enables or disables the "Find Candidates" button based on whether a job description is entered and resumes are uploaded.
*   **Inputs:** `self`.
*   **Outputs:** None. Configures the state of `self.find_candidates_button`.
*   **Workflow:**
    1.  Retrieves the content of `self.jd_text_area`.
    2.  Checks if `jd_content` is not empty and `self.resume_paths` is not empty.
    3.  Sets the state of `self.find_candidates_button` to `tk.NORMAL` (enabled) or `tk.DISABLED` accordingly.

#### `SemanticResumeMatcherApp.find_candidates(self)`

*   **Purpose:** Initiates the semantic matching process, displays results, and logs.
*   **Inputs:** `self`.
*   **Outputs:** None. Updates GUI elements with results and status.
*   **Workflow:**
    1.  Performs input validation for the job description.
    2.  Updates `self.status_label` to "Processing..." and clears `self.results_listbox`.
    3.  Records the `start_time` for performance tracking.
    4.  Instantiates `MatchingEngine`.
    5.  Calls `engine.match_resumes_to_job_description_text()` with the job description content and resume paths to get results and logs.
    6.  Records the `end_time` and calculates `duration`.
    7.  Clears `self.log_text_area`, enables it, inserts the received logs and processing time, then disables it.
    8.  Checks if `results` indicate an error (string type). If so, displays an error message.
    9.  If `results` are valid, updates `self.status_label` with "Matching complete" and duration, and populates `self.results_listbox` with ranked candidates.
    10. If no matching candidates are found, updates `self.status_label` and shows an info message.
    11. Re-enables `self.find_candidates_button`.

#### `SemanticResumeMatcherApp.copy_selected_result(self, event=None)`

*   **Purpose:** Copies the text of selected items from the results listbox to the clipboard.
*   **Inputs:** `self`, `event` (optional, for key binding).
*   **Outputs:** None. Copies text to clipboard and shows an info message.
*   **Workflow:**
    1.  Gets the indices of selected items in `self.results_listbox`.
    2.  If no items are selected, returns.
    3.  Retrieves the text for each selected item.
    4.  Clears the clipboard and appends the joined selected text.
    5.  Displays a "Copied to clipboard" info message.

### `matching_engine.py`

#### `MatchingEngine.__init__(self)`

*   **Purpose:** Initializes the semantic matching engine, including the SentenceTransformer model and FAISS index.
*   **Inputs:** `self`.
*   **Outputs:** None. Initializes instance variables.
*   **Workflow:**
    1.  Initializes `self.model` with `SentenceTransformer('all-MiniLM-L6-v2')`.
    2.  Sets `self.faiss_index` to `None`.
    3.  Initializes empty lists for `self.resume_texts`, `self.resume_filenames`, and `self.calculation_logs`.

#### `MatchingEngine.extract_text_from_pdf(self, pdf_path)`

*   **Purpose:** Extracts text content from a PDF file.
*   **Inputs:**
    *   `pdf_path`: The absolute path to the PDF file.
*   **Outputs:** The extracted text as a string, or an empty string if an error occurs.
*   **Workflow:**
    1.  Opens the PDF file in binary read mode.
    2.  Creates a `PdfReader` object.
    3.  Iterates through each page, extracting text and concatenating it.
    4.  Includes error handling for file operations.

#### `MatchingEngine.extract_text_from_docx(self, docx_path)`

*   **Purpose:** Extracts text content from a DOCX file.
*   **Inputs:**
    *   `docx_path`: The absolute path to the DOCX file.
*   **Outputs:** The extracted text as a string, or an empty string if an error occurs.
*   **Workflow:**
    1.  Creates a `Document` object from the DOCX file.
    2.  Iterates through each paragraph, extracting text and concatenating it.
    3.  Includes error handling for file operations.

#### `MatchingEngine.get_text_from_file(self, file_path)`

*   **Purpose:** A utility function to determine file type and call the appropriate text extraction method.
*   **Inputs:**
    *   `file_path`: The absolute path to the file (PDF or DOCX).
*   **Outputs:** The extracted text as a string.
*   **Workflow:**
    1.  Determines the file extension.
    2.  Calls `self.extract_text_from_pdf()` for `.pdf` files.
    3.  Calls `self.extract_text_from_docx()` for `.docx` files.
    4.  Returns an empty string for unsupported file types.

#### `MatchingEngine.process_job_description(self, jd_path)`

*   **Purpose:** Extracts text from a job description file and generates its semantic embedding.
*   **Inputs:**
    *   `jd_path`: The absolute path to the job description file.
*   **Outputs:** The job description's embedding (NumPy array) or `None` if processing fails.
*   **Workflow:**
    1.  Extracts text from the JD file using `self.get_text_from_file()`.
    2.  If text is extracted, encodes it into an embedding using `self.model.encode()`.

#### `MatchingEngine.process_resumes(self, resume_paths)`

*   **Purpose:** Extracts text from multiple resume files, generates their semantic embeddings, and builds the FAISS index.
*   **Inputs:**
    *   `resume_paths`: A list of absolute paths to resume files.
*   **Outputs:** `True` if resumes are processed successfully, `None` otherwise.
*   **Workflow:**
    1.  Clears `self.resume_texts`, `self.resume_filenames`, and `resume_embeddings`.
    2.  Iterates through each `resume_path`:
        *   Extracts text using `self.get_text_from_file()`.
        *   If text is extracted, appends it to `self.resume_texts`, the filename to `self.resume_filenames`, and its embedding to `resume_embeddings`.
    3.  If no embeddings are generated, returns `None`.
    4.  Converts `resume_embeddings` to a NumPy array (`resume_embeddings_np`).
    5.  Initializes `self.faiss_index` (an `IndexFlatL2`) with the embedding dimension.
    6.  Adds `resume_embeddings_np` to `self.faiss_index`.

#### `MatchingEngine.find_top_candidates(self, jd_embedding, k=10)`

*   **Purpose:** Finds the top `k` most semantically similar resumes to a given job description embedding using the FAISS index.
*   **Inputs:**
    *   `jd_embedding`: The semantic embedding of the job description.
    *   `k`: The number of top candidates to retrieve (default: 10).
*   **Outputs:** A list of dictionaries, each containing 'filename' and 'score' for ranked candidates.
*   **Workflow:**
    1.  Checks if `self.faiss_index` is initialized and `self.resume_filenames` is not empty.
    2.  Limits `actual_k` to the minimum of `k` and the number of available resumes.
    3.  Performs a FAISS search using `self.faiss_index.search()` to get distances (`D`) and indices (`I`) of top candidates.
    4.  Initializes `ranked_results` and `seen_filenames` (a set to track unique filenames).
    5.  Iterates through the search results (`I[0]`):
        *   Retrieves `resume_index`.
        *   Skips invalid indices (`-1`).
        *   Retrieves `filename`.
        *   Skips if `filename` is already in `seen_filenames` (ensuring uniqueness).
        *   Calculates `jd_embedding_norm` and `resume_embedding_norm`.
        *   Calculates `similarity_score` using the dot product (cosine similarity).
        *   Appends a human-readable `log_entry` to `self.calculation_logs` explaining the similarity calculation.
        *   Appends a dictionary with 'filename' and 'score' to `ranked_results`.
        *   Adds `filename` to `seen_filenames`.
    6.  Sorts `ranked_results` by 'score' in descending order.

#### `MatchingEngine.match_resumes_to_job_description(self, jd_path, resume_paths)`

*   **Purpose:** Orchestrates the entire matching process when a job description is provided as a file path.
*   **Inputs:**
    *   `jd_path`: Absolute path to the job description file.
    *   `resume_paths`: List of absolute paths to resume files.
*   **Outputs:** A list of ranked candidate dictionaries, or an error string.
*   **Workflow:**
    1.  Resets `self.faiss_index`, `self.resume_texts`, and `self.resume_filenames`.
    2.  Processes the job description file to get `jd_embedding`. Returns an error if processing fails.
    3.  Processes resume files to build the FAISS index. Returns an error if processing fails.
    4.  Finds top candidates using `self.find_top_candidates()`.

#### `MatchingEngine.process_job_description_text(self, jd_text)`

*   **Purpose:** Generates a semantic embedding for a job description provided as raw text.
*   **Inputs:**
    *   `jd_text`: The job description content as a string.
*   **Outputs:** The job description's embedding (NumPy array) or `None` if processing fails.
*   **Workflow:**
    1.  If `jd_text` is empty, returns `None`.
    2.  Encodes `jd_text` into an embedding using `self.model.encode()`.

#### `MatchingEngine.match_resumes_to_job_description_text(self, jd_text, resume_paths)`

*   **Purpose:** Orchestrates the entire matching process when a job description is provided as raw text. This is the primary entry point used by the GUI.
*   **Inputs:**
    *   `jd_text`: The job description content as a string.
    *   `resume_paths`: List of absolute paths to resume files.
*   **Outputs:** A tuple containing a list of ranked candidate dictionaries and a list of calculation logs, or an error string and an empty list of logs.
*   **Workflow:**
    1.  Resets `self.faiss_index`, `self.resume_texts`, `self.resume_filenames`, and `self.calculation_logs`.
    2.  Processes the job description text to get `jd_embedding`. Returns an error and empty logs if processing fails.
    3.  Processes resume files to build the FAISS index. Returns an error and empty logs if processing fails.
    4.  Finds top candidates using `self.find_top_candidates()`.
    5.  Returns the ranked results and the calculation logs.
