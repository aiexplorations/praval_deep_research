# PDF Upload Feature Plan

## Objective

Enable users to upload PDF research papers (single or bulk) directly into the knowledge base, bypassing ArXiv search. Uploaded PDFs should go through the same indexing pipeline as ArXiv papers, ensuring consistency in search and Q&A capabilities.

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend      │     │   Backend API    │     │     Storage     │
│  Upload UI      │────▶│  /upload/papers  │────▶│     MinIO       │
│ (drag-drop)     │     │  (multipart)     │     │   (PDF files)   │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                                 │ Emit Event
                                 ▼
                        ┌──────────────────┐
                        │  papers_found    │
                        │  (Praval Spore)  │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Document         │
                        │ Processor Agent  │
                        └────────┬─────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
             │  Qdrant  │ │  Vajra   │ │  MinIO   │
             │ (vectors)│ │ (BM25)   │ │ (text)   │
             └──────────┘ └──────────┘ └──────────┘
```

---

## Key Design Decisions

1. **Reuse Existing Pipeline**: The document processor already handles PDF processing, chunking, and embedding generation. We emit a `papers_found` event to trigger it.

2. **Paper ID Generation**: Uploaded PDFs need unique IDs. Use format: `upload_<timestamp>_<hash>` to distinguish from ArXiv papers.

3. **Metadata Extraction**: Extract title/authors from PDF metadata or first page. Allow user override in upload form.

4. **Incremental Vajra Indexing**: Add real-time Vajra BM25 indexing in the document processor (currently only builds from Qdrant at startup).

5. **Bulk Upload**: Support multiple files in single request with per-file progress tracking.

---

## Implementation Plan

### Phase 1: Backend Upload Endpoint

**New File:** `src/agentic_research/api/routes/upload.py`

```python
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional
import hashlib
import time

router = APIRouter(prefix="/upload", tags=["upload"])

@router.post("/papers")
async def upload_papers(
    files: List[UploadFile] = File(...),
    titles: Optional[List[str]] = Form(None),  # Optional overrides
    authors: Optional[List[str]] = Form(None),
    categories: Optional[List[str]] = Form(None),
) -> Dict[str, Any]:
    """
    Upload one or more PDF papers for indexing.

    Args:
        files: PDF files to upload
        titles: Optional title overrides (one per file)
        authors: Optional author overrides (comma-separated per file)
        categories: Optional category overrides (comma-separated per file)

    Returns:
        Upload status with generated paper IDs
    """
```

**Request/Response Models:**

```python
class UploadPaperResponse(BaseModel):
    paper_id: str
    title: str
    filename: str
    status: str  # "queued", "processing", "complete", "error"

class UploadBatchResponse(BaseModel):
    batch_id: str
    papers: List[UploadPaperResponse]
    total_files: int
    queued: int
```

### Phase 2: PDF Metadata Extraction

**New File:** `src/processors/pdf_metadata.py`

```python
from pypdf import PdfReader
from typing import Dict, Any, Optional

def extract_pdf_metadata(pdf_data: bytes) -> Dict[str, Any]:
    """
    Extract metadata from PDF file.

    Attempts to extract:
    - Title (from PDF metadata or first heading)
    - Authors (from PDF metadata)
    - Creation date
    - Subject/keywords

    Returns:
        Dict with extracted metadata, empty strings for missing fields
    """
    reader = PdfReader(io.BytesIO(pdf_data))
    metadata = reader.metadata or {}

    return {
        "title": metadata.get("/Title", "") or _extract_title_from_text(reader),
        "authors": _parse_authors(metadata.get("/Author", "")),
        "created_date": metadata.get("/CreationDate", ""),
        "subject": metadata.get("/Subject", ""),
        "keywords": metadata.get("/Keywords", ""),
    }

def _extract_title_from_text(reader: PdfReader) -> str:
    """Extract title from first page text (first large heading)."""
    # Use LLM to identify title from first page
    first_page = reader.pages[0].extract_text()[:2000]
    return chat(f"Extract the paper title from this text:\n{first_page}")
```

### Phase 3: MinIO Storage for Uploads

**Modify:** `src/agentic_research/storage/minio_client.py`

Add method for uploaded PDFs:

```python
def upload_user_pdf(self, paper_id: str, pdf_data: bytes, filename: str) -> str:
    """
    Upload user-provided PDF to MinIO.

    Args:
        paper_id: Generated paper ID (upload_xxx format)
        pdf_data: Raw PDF bytes
        filename: Original filename for reference

    Returns:
        MinIO object path
    """
    object_name = f"uploads/{paper_id}/{filename}"
    # ... upload logic
    return object_name
```

### Phase 4: Event Emission

After storing PDF in MinIO, emit `papers_found` event:

```python
from praval import broadcast

# After successful upload and metadata extraction
papers_data = []
for paper_id, metadata in uploaded_papers:
    papers_data.append({
        "arxiv_id": paper_id,  # Use paper_id in arxiv_id field for compatibility
        "title": metadata["title"],
        "authors": metadata["authors"],
        "abstract": "",  # No abstract for uploads
        "categories": metadata.get("categories", ["uploaded"]),
        "published_date": metadata.get("created_date", ""),
        "pdf_source": "upload",  # Flag to indicate upload vs ArXiv
        "pdf_path": minio_path,  # Pre-stored path
    })

# Emit event to trigger document processor
broadcast({
    "type": "papers_found",
    "knowledge": {
        "papers": papers_data,
        "original_query": "user_upload",
        "search_metadata": {
            "domain": "upload",
            "source": "user_upload",
            "batch_id": batch_id,
        }
    }
})
```

### Phase 5: Document Processor Updates

**Modify:** `src/agents/research/document_processor.py`

Add handling for pre-uploaded PDFs:

```python
# In the processing loop
if paper.get("pdf_source") == "upload" and paper.get("pdf_path"):
    # PDF already in MinIO, just download for processing
    pdf_data = minio_client.download_pdf(paper["pdf_path"])
    pdf_path = paper["pdf_path"]
else:
    # Normal ArXiv download flow
    pdf_data = pdf_processor.download_from_arxiv_sync(arxiv_id)
    pdf_path = minio_client.upload_pdf(arxiv_id, pdf_data)
```

Add incremental Vajra indexing after Qdrant storage:

```python
# After storing in Qdrant (around line 239)
from agentic_research.storage.paper_index import get_paper_index

# Index to Vajra BM25 for keyword search
try:
    paper_index = get_paper_index()
    paper_index.index_paper(
        paper_id=arxiv_id,
        title=paper.get('title', ''),
        chunks=[c.get('chunk_text', '') for c in chunks_with_embeddings],
        authors=paper.get('authors', []),
        categories=paper.get('categories', []),
        abstract=paper.get('abstract', ''),
        published_date=paper.get('published_date', ''),
    )
    logger.info(f"Indexed {arxiv_id} to Vajra BM25")
except Exception as e:
    logger.warning(f"Vajra indexing failed for {arxiv_id}: {e}")
    # Non-fatal - Qdrant indexing succeeded
```

### Phase 6: Frontend Upload Component

**New File:** `frontend-new/src/components/upload/PaperUpload.tsx`

```typescript
interface PaperUploadProps {
  onUploadComplete: (papers: UploadedPaper[]) => void;
}

export const PaperUpload: React.FC<PaperUploadProps> = ({ onUploadComplete }) => {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<Record<string, number>>({});

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files)
      .filter(f => f.type === 'application/pdf');
    setFiles(prev => [...prev, ...droppedFiles]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    files.forEach(f => formData.append('files', f));

    const response = await api.uploadPapers(formData);
    onUploadComplete(response.papers);
  };

  return (
    <div
      className="upload-zone"
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
    >
      {/* Drop zone UI */}
      {/* File list with metadata edit */}
      {/* Upload button with progress */}
    </div>
  );
};
```

**New File:** `frontend-new/src/components/upload/UploadProgress.tsx`

```typescript
// SSE-driven progress tracking
export const UploadProgress: React.FC<{ batchId: string }> = ({ batchId }) => {
  // Subscribe to indexing_progress events
  // Show per-paper progress
  // Display errors if any
};
```

### Phase 7: API Client Extension

**Modify:** `frontend-new/src/services/api/client.ts`

```typescript
async uploadPapers(formData: FormData): Promise<UploadBatchResponse> {
  const response = await fetch(`${this.baseUrl}/upload/papers`, {
    method: 'POST',
    body: formData,
  });
  return response.json();
}
```

### Phase 8: Knowledge Base Page Integration

**Modify:** `frontend-new/src/pages/KnowledgeBase.tsx`

Add upload button and modal:

```typescript
// In the page header
<Button onClick={() => setShowUploadModal(true)}>
  <Upload className="w-4 h-4 mr-2" />
  Upload Papers
</Button>

{showUploadModal && (
  <Modal onClose={() => setShowUploadModal(false)}>
    <PaperUpload onUploadComplete={handleUploadComplete} />
  </Modal>
)}
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/agentic_research/api/routes/upload.py` | Upload endpoint |
| `src/processors/pdf_metadata.py` | PDF metadata extraction |
| `frontend-new/src/components/upload/PaperUpload.tsx` | Upload UI component |
| `frontend-new/src/components/upload/UploadProgress.tsx` | Progress tracking |
| `frontend-new/src/types/upload.ts` | TypeScript types |

## Files to Modify

| File | Changes |
|------|---------|
| `src/agentic_research/api/main.py` | Register upload router |
| `src/agentic_research/storage/minio_client.py` | Add upload methods |
| `src/agents/research/document_processor.py` | Handle pre-uploaded PDFs + Vajra indexing |
| `frontend-new/src/services/api/client.ts` | Add uploadPapers method |
| `frontend-new/src/pages/KnowledgeBase.tsx` | Add upload button |

---

## Event Flow

```
1. User drops PDF files in upload zone
   └─▶ Frontend validates files (PDF type, size limits)

2. User clicks "Upload"
   └─▶ POST /upload/papers with FormData

3. Backend processes each file:
   a. Generate paper_id: upload_<timestamp>_<hash>
   b. Extract metadata from PDF
   c. Store PDF in MinIO (uploads/<paper_id>/)
   d. Add to papers_data list

4. Backend emits papers_found event:
   └─▶ broadcast({"type": "papers_found", "knowledge": {...}})

5. Document Processor Agent triggered:
   a. Detects pdf_source == "upload"
   b. Downloads PDF from MinIO (already stored)
   c. Extracts text, chunks
   d. Generates embeddings
   e. Stores in Qdrant
   f. Indexes to Vajra BM25 (NEW)
   g. Emits SSE progress events

6. Frontend receives SSE events:
   └─▶ Updates progress UI per paper

7. Indexing complete:
   └─▶ Papers appear in Knowledge Base
   └─▶ Available for hybrid search and Q&A
```

---

## Verification Checklist

- [ ] Single PDF upload works
- [ ] Bulk upload (5+ files) works with progress
- [ ] Metadata extraction from PDF
- [ ] Manual metadata override in upload form
- [ ] PDF stored in MinIO under uploads/
- [ ] Document processor triggered via event
- [ ] Qdrant vectors created
- [ ] Vajra BM25 index updated
- [ ] Paper appears in Knowledge Base list
- [ ] Paper searchable via hybrid search
- [ ] Paper available for Chat with Papers
- [ ] SSE progress events received by frontend
- [ ] Error handling for invalid/corrupt PDFs

---

## UI Mockup

```
┌─────────────────────────────────────────────────────────────────┐
│  Knowledge Base                              [+ Upload Papers]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                                                           │  │
│  │      ┌─────────────────────────────────────────────┐      │  │
│  │      │                                             │      │  │
│  │      │         📄 Drop PDF files here              │      │  │
│  │      │                                             │      │  │
│  │      │            or click to browse               │      │  │
│  │      │                                             │      │  │
│  │      └─────────────────────────────────────────────┘      │  │
│  │                                                           │  │
│  │  Selected Files:                                          │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ 📄 attention-is-all-you-need.pdf     [Edit] [✕]     │  │  │
│  │  │    Title: Attention Is All You Need (auto-detected) │  │  │
│  │  ├─────────────────────────────────────────────────────┤  │  │
│  │  │ 📄 bert-paper.pdf                    [Edit] [✕]     │  │  │
│  │  │    Title: BERT: Pre-training of Deep... (detected)  │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │                    [Upload 2 Papers]                      │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Upload Progress:                                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ attention-is-all-you-need.pdf  ████████████░░░░  75%       ││
│  │ Processing: Generating embeddings for chunk 45/60...       ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │ bert-paper.pdf                 ░░░░░░░░░░░░░░░░  Queued    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Security Considerations

1. **File Validation**: Check MIME type and magic bytes, not just extension
2. **Size Limits**: Max 50MB per file, max 10 files per batch
3. **Sanitization**: Sanitize filenames before storage
4. **Rate Limiting**: Limit uploads per user/IP per hour
5. **Virus Scanning**: Consider ClamAV integration for production

---

## Future Enhancements

1. **URL Upload**: Accept URLs to PDFs (download and process)
2. **Folder Watch**: Watch a folder for new PDFs (desktop mode)
3. **Duplicate Detection**: Check if paper already indexed (by content hash)
4. **OCR Support**: Handle scanned PDFs with Tesseract
5. **Batch Operations**: Export/import entire knowledge bases

---

## Dependencies

No new dependencies required. Uses existing:
- `pypdf` for PDF reading
- `praval` for event broadcasting
- MinIO client for storage
- Existing document processor pipeline
