"""
Minimal Streamlit chatbot frontend that connects to your existing backend.

Backend contract expected (adjust BASE_URL and paths below to match your API):

1) POST {BASE_URL}/chat
   request:  {"message": "<user text>"}
   response: {"reply": "<assistant text>"}

2) POST {BASE_URL}/upload   (multipart/form-data, field name "file")
   - backend should: save the PDF, embed it, store it in ChromaDB
   response: {"status": "ok", "filename": "<name>", "chunks": <int, optional>}

3) GET {BASE_URL}/documents   (OPTIONAL but recommended)
   - backend should return the list of PDFs currently stored in ChromaDB,
     read from Chroma's collection metadata (not from a temp folder), so the
     list is always the source of truth and survives app restarts.
   response: {"documents": [{"filename": "x.pdf", "chunks": 12}, ...]}

If you don't have endpoint 3 yet, the app falls back to just remembering
files uploaded in this browser session (st.session_state), which works but
resets if you refresh the page or open a new session.
"""

import streamlit as st
import requests

# ---------------------------------------------------------------------------
# CONFIG — change this to match where your backend is running
# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{BASE_URL}/ask"
UPLOAD_ENDPOINT = f"{BASE_URL}/upload-pdfs"
DOCUMENTS_ENDPOINT = f"{BASE_URL}/list-docs"  # optional, see note above

st.set_page_config(page_title="My Chatbot", page_icon="💬", layout="wide")

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []  # fallback list if /documents isn't available


def fetch_documents_from_backend():
    """Pull the authoritative list of stored PDFs from the backend/ChromaDB."""
    try:
        resp = requests.get(DOCUMENTS_ENDPOINT, timeout=10)
        if resp.ok:
            return resp.json().get("documents", [])
    except requests.exceptions.RequestException:
        pass
    return None  # signals "couldn't reach endpoint" -> fall back to session list


# ---------------------------------------------------------------------------
# LAYOUT: two columns — chat on the left, documents/upload on the right
# ---------------------------------------------------------------------------
chat_col, side_col = st.columns([2, 1])

# ---------------------------------------------------------------------------
# CHAT SECTION
# ---------------------------------------------------------------------------
with chat_col:
    st.subheader("💬 Chat")

    chat_box = st.container(height=500)
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("Type a message...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.get(CHAT_ENDPOINT, json={"question": user_input})
                    if resp.ok:
                        reply = resp.json().get("answer", "(empty response from backend)")
                    else:
                        reply = f"⚠️ Backend returned status {resp.status_code}: {resp.text}"
                except requests.exceptions.RequestException as e:
                    reply = f"⚠️ Could not reach backend at {CHAT_ENDPOINT}\n\n`{e}`"
            
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# ---------------------------------------------------------------------------
# UPLOAD + DOCUMENT LIST SECTION
# ---------------------------------------------------------------------------
with side_col:
    st.subheader("📄 Upload PDFs")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Upload & Embed", use_container_width=True):
            with st.spinner("Uploading and embedding into ChromaDB..."):
                try:
                   # Change "file" to "files"
                    files = {"files": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    resp = requests.post(UPLOAD_ENDPOINT, files=files, timeout=120)
                    if resp.ok:
                        data = resp.json()
                        st.success(f"✅ {data.get('filename', uploaded_file.name)} uploaded and embedded.")
                        # keep a local fallback record too
                        st.session_state.uploaded_docs.append(
                            {"filename": data.get("filename", uploaded_file.name),
                             "chunks": data.get("chunks", "—")}
                        )
                    else:
                        st.error(f"Backend returned status {resp.status_code}: {resp.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach backend at {UPLOAD_ENDPOINT}\n\n{e}")

    st.divider()
    st.subheader("📚 Stored Documents")

    col_a, col_b = st.columns([3, 1])
    with col_b:
        refresh = st.button("🔄", help="Refresh list from backend")

    backend_docs = fetch_documents_from_backend()

    if backend_docs is not None:
        docs_to_show = backend_docs
        source_note = "from ChromaDB (live)"
    else:
        docs_to_show = st.session_state.uploaded_docs
        source_note = "from this session only — add a GET /documents endpoint for a persistent list"

    st.caption(source_note)

    if docs_to_show:
        for doc in docs_to_show:
            name = doc
            # chunks = doc.get("chunks", "—")
            st.markdown(f"- **{name}**  \n ")
    else:
        st.info("No documents uploaded yet.")
