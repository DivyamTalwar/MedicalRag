"""
PDF - Simple PDF extraction for medical documents
"""
import pdfplumber
from typing import List, Dict, Any
from pathlib import Path

class PDFExtractor:
    def __init__(self):
        print("[OK] PDF extractor initialized")
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        if not pdf_path or not Path(pdf_path).exists():
            raise ValueError(f"PDF file not found: {pdf_path}")
        
        print(f"[INFO] Extracting text from: {Path(pdf_path).name}")
        
        try:
            text_parts = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"Page {i}:\n{page_text}")
                
                total_text = "\n\n".join(text_parts)
                print(f"[OK] Extracted {len(total_text)} characters from {len(pdf.pages)} pages")
                return total_text
        
        except Exception as e:
            print(f"[ERROR] Failed to extract PDF: {e}")
            raise Exception(f"PDF extraction failed: {e}")
    
    def extract_chunks(self, pdf_path: str, chunk_size: int = 500) -> List[str]:
        """Extract text and split into chunks"""
        full_text = self.extract_text(pdf_path)
        
        if not full_text.strip():
            return []
        
        # Simple chunking by words
        words = full_text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk.strip())
        
        print(f"[OK] Created {len(chunks)} chunks of ~{chunk_size} words each")
        return chunks
    
    def extract_and_index(self, pdf_path: str, search_service) -> bool:
        """Extract PDF and add to search index"""
        try:
            # Extract chunks
            chunks = self.extract_chunks(pdf_path)
            
            if not chunks:
                print("[WARNING] No text extracted from PDF")
                return False
            
            # Create metadata for each chunk
            pdf_name = Path(pdf_path).stem
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                metadatas.append({
                    "source": pdf_name,
                    "chunk_id": i,
                    "type": "pdf_chunk",
                    "file_path": pdf_path
                })
            
            # Add to search index
            success = search_service.add_documents(chunks, metadatas)
            
            if success:
                print(f"[OK] Successfully indexed {len(chunks)} chunks from {pdf_name}")
            else:
                print(f"[ERROR] Failed to index chunks from {pdf_name}")
            
            return success
            
        except Exception as e:
            print(f"[ERROR] Failed to extract and index PDF: {e}")
            return False

# Test function
def test_pdf():
    """Test PDF extraction"""
    extractor = PDFExtractor()
    
    # Note: This is a test function - in practice you would provide actual PDF paths
    print("[INFO] PDF extractor ready for use")
    print("[INFO] To test with actual PDF:")
    print("  extractor.extract_text('path/to/medical.pdf')")
    print("  extractor.extract_chunks('path/to/medical.pdf')")
    
    return extractor

if __name__ == "__main__":
    test_pdf()