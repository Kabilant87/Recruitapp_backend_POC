import os
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class MatchingEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.faiss_index = None
        self.resume_texts = []
        self.resume_filenames = []
        self.calculation_logs = [] # New: To store detailed calculation logs


    def extract_text_from_pdf(self, pdf_path):
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
        return text

    def extract_text_from_docx(self, docx_path):
        text = ""
        try:
            document = Document(docx_path)
            for paragraph in document.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error extracting text from DOCX {docx_path}: {e}")
        return text

    def get_text_from_file(self, file_path):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return self.extract_text_from_docx(file_path)
        else:
            return ""

    def process_job_description(self, jd_path):
        jd_text = self.get_text_from_file(jd_path)
        if not jd_text:
            return None
        jd_embedding = self.model.encode([jd_text])[0]
        return jd_embedding

    def process_resumes(self, resume_paths):
        self.resume_texts = []
        self.resume_filenames = []
        resume_embeddings = []

        for path in resume_paths:
            text = self.get_text_from_file(path)
            if text:
                self.resume_texts.append(text)
                self.resume_filenames.append(os.path.basename(path))
                resume_embeddings.append(self.model.encode([text])[0])

        if not resume_embeddings:
            return None

        # Convert list of embeddings to a numpy array
        resume_embeddings_np = np.array(resume_embeddings).astype('float32')
        self.resume_embeddings_np = resume_embeddings_np # Store embeddings for later use

        # Build FAISS index
        dimension = resume_embeddings_np.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension) # L2 distance for similarity
        self.faiss_index.add(resume_embeddings_np) # Add resume embeddings to the FAISS index

        return True

    def find_top_candidates(self, jd_embedding, k=10):
        if self.faiss_index is None or not self.resume_filenames: # Also check if resume_filenames is empty
            return []

        # Limit k to the actual number of resumes available
        actual_k = min(k, len(self.resume_filenames))
        if actual_k == 0: # No resumes to search
            return []

        # Reshape jd_embedding for FAISS search
        D, I = self.faiss_index.search(np.array([jd_embedding]).astype('float32'), actual_k)

        ranked_results = []
        seen_filenames = set() # To track unique filenames

        for i in range(len(I[0])):
            resume_index = I[0][i]

            # Skip invalid indices (FAISS returns -1 for non-existent results if k > num_vectors)
            if resume_index == -1:
                continue

            filename = self.resume_filenames[resume_index]

            # Skip if this filename has already been added to ranked_results
            if filename in seen_filenames:
                continue
            
            distance = D[0][i]
            # FAISS returns L2 distance, convert to similarity score (0 to 1)
            # A common way is 1 / (1 + distance), or using cosine similarity if embeddings are normalized
            # For now, let's use a simple inverse of distance, or just the distance itself and sort by it
            # For L2, smaller distance means more similar. So, 1 - normalized_distance might be better.
            # Let's just return distance for now and let GUI format.
            # Or, if embeddings are normalized, dot product is cosine similarity.
            # We need to re-calculate cosine similarity from embeddings if we want 0-1 score.
            # For now, let's just return the distance and sort by it (ascending for L2 distance).
            # Or, let's calculate cosine similarity for better interpretation.

            # Recalculate cosine similarity for a 0-1 score
            jd_embedding_norm = jd_embedding / np.linalg.norm(jd_embedding)
            resume_embedding_norm = self.resume_embeddings_np[resume_index] / np.linalg.norm(self.resume_embeddings_np[resume_index])
            similarity_score = np.dot(jd_embedding_norm, resume_embedding_norm)

            # Log the calculation details
            # NOTE: Including full JD and Resume text can make logs very verbose.
            # Log the calculation details
            log_entry = f"--- Candidate: {self.resume_filenames[resume_index]} ---\n" \
                        f"  Similarity Calculation: Cosine Similarity\n" \
                        f"  Job Description and Resume embeddings were compared.\n" \
                        f"  A higher score indicates a stronger semantic match.\n" \
                        f"  Calculated Match Score: {similarity_score:.4f}\n"
            self.calculation_logs.append(log_entry)

            ranked_results.append({
                "filename": self.resume_filenames[resume_index],
                "score": float(similarity_score) # Ensure it's a standard float
            })
            seen_filenames.add(filename) # Add filename to seen_filenames after adding to ranked_results
        
        # Sort by score in descending order (higher score means more similar)
        ranked_results.sort(key=lambda x: x['score'], reverse=True)

        return ranked_results

    def match_resumes_to_job_description(self, jd_path, resume_paths):
        self.faiss_index = None # Reset index for new search
        self.resume_texts = []
        self.resume_filenames = []

        jd_embedding = self.process_job_description(jd_path)
        if jd_embedding is None:
            return "Error: Could not process job description."

        if not self.process_resumes(resume_paths):
            return "Error: No resumes processed or embeddings generated."

        results = self.find_top_candidates(jd_embedding)
        return results

    def process_job_description_text(self, jd_text):
        if not jd_text:
            return None
        jd_embedding = self.model.encode([jd_text])[0]
        return jd_embedding

    def match_resumes_to_job_description_text(self, jd_text, resume_paths):
        self.faiss_index = None # Reset index for new search
        self.resume_texts = []
        self.resume_filenames = []

        self.calculation_logs = [] # Reset logs for new search

        jd_embedding = self.process_job_description_text(jd_text)
        if jd_embedding is None:
            return "Error: Could not process job description text.", [] # Return empty logs on error

        if not self.process_resumes(resume_paths):
            return "Error: No resumes processed or embeddings generated.", [] # Return empty logs on error

        results = self.find_top_candidates(jd_embedding)
        return results, self.calculation_logs # Return both results and logs
