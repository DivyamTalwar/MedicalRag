import os
from pinecone import Pinecone
from dotenv import load_dotenv
import json

load_dotenv()

def fetch_pdf_metadata(pdf_name="CIVIE-RIS.pdf"):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    PARENT_INDEX_NAME = "parent"
    CHILD_INDEX_NAME = "children"
    
    all_metadata = {
        "parent_chunks": [],
        "child_chunks": [],
        "summary": {}
    }
    
    print(f"Fetching metadata for PDF: {pdf_name}")
    
    try:
        parent_index = pc.Index(PARENT_INDEX_NAME)
        
        parent_results = parent_index.query(
            vector=[0.0] * 1024,
            filter={"pdf_name": pdf_name},
            top_k=10000,
            include_metadata=True,
            include_values=False
        )
        
        print(f"Found {len(parent_results['matches'])} parent chunks")
        
        for i, match in enumerate(parent_results['matches']):
            metadata = match['metadata']
            all_metadata["parent_chunks"].append({
                "chunk_index": i + 1,
                "chunk_id": match['id'],
                "metadata": metadata
            })
            
    except Exception as e:
        print(f"Error fetching parent chunks: {e}")
    
    try:
        child_index = pc.Index(CHILD_INDEX_NAME)
        
        child_results = child_index.query(
            vector=[0.0] * 1024,
            filter={"pdf_name": pdf_name},
            top_k=10000,
            include_metadata=True,
            include_values=False
        )
        
        print(f"Found {len(child_results['matches'])} child chunks")
        
        for i, match in enumerate(child_results['matches']):
            metadata = match['metadata']
            all_metadata["child_chunks"].append({
                "chunk_index": i + 1, 
                "chunk_id": match['id'],
                "metadata": metadata
            })
            
    except Exception as e:
        print(f"Error fetching child chunks: {e}")
    
    all_metadata["summary"] = {
        "total_parent_chunks": len(all_metadata["parent_chunks"]),
        "total_child_chunks": len(all_metadata["child_chunks"]),
        "total_chunks": len(all_metadata["parent_chunks"]) + len(all_metadata["child_chunks"])
    }
    
    print("\nSUMMARY:")
    print(f"Parent chunks: {all_metadata['summary']['total_parent_chunks']}")
    print(f"Child chunks: {all_metadata['summary']['total_child_chunks']}")
    print(f"Total chunks: {all_metadata['summary']['total_chunks']}")
    
    return all_metadata

def save_metadata_to_file(metadata_dict, filename="civie_ris_metadata.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    print(f"\nAll metadata saved to: {filename}")

if __name__ == "__main__":
    metadata = fetch_pdf_metadata("CIVIE-RIS.pdf")
    
    save_metadata_to_file(metadata)
    
    print(f"\nComplete.")
