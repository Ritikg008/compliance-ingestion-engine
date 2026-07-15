import streamlit as st
import requests
import time

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Compliance Ingestion Engine",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/law.png", width=64)
st.sidebar.title("⚖️ Compliance Engine")
st.sidebar.markdown("Multi-modal AI compliance auditing for YouTube videos.")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🎬 Ingest & Audit", "🔍 Query Videos"],
)
st.sidebar.markdown("---")

# Health check indicator in sidebar
try:
    r = requests.get(f"{API_BASE}/health", timeout=3)
    if r.status_code == 200:
        st.sidebar.success("🟢 API is online")
    else:
        st.sidebar.error("🔴 API returned error")
except Exception:
    st.sidebar.error("🔴 API is offline — start uvicorn first")


# ---------------------------------------------------------------------------
# Page 1: Ingest & Audit
# ---------------------------------------------------------------------------
if page == "🎬 Ingest & Audit":
    st.title("🎬 Ingest & Audit a YouTube Video")
    st.markdown(
        "Paste any YouTube URL below. The engine will download it, "
        "transcribe the audio, extract on-screen text via OCR, index "
        "everything into a vector database, and run an AI compliance audit."
    )

    url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    if st.button("🚀 Run Pipeline", disabled=not url):
        with st.status("Running ingestion pipeline...", expanded=True) as status:
            st.write("📥 Downloading video...")
            time.sleep(0.5)
            st.write("🎙️ Transcribing audio with Whisper...")
            time.sleep(0.5)
            st.write("🔍 Extracting frames and running OCR...")
            time.sleep(0.5)
            st.write("✂️  Chunking and indexing into Qdrant...")
            time.sleep(0.5)
            st.write("⚖️  Running compliance audit with Groq LLM...")

            try:
                response = requests.post(
                    f"{API_BASE}/ingest",
                    json={"url": url},
                    timeout=300,
                )

                if response.status_code == 200:
                    data = response.json()
                    status.update(
                        label="✅ Pipeline complete!",
                        state="complete",
                        expanded=False,
                    )

                    # --- Video info ---
                    st.markdown("---")
                    st.subheader("📹 Video Info")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Video ID", data["video_id"])
                    col2.metric("Chunks Indexed", data["chunks_indexed"])
                    col3.metric(
                        "Sources",
                        "Transcript + OCR",
                    )
                    st.markdown(f"**Title:** {data['video_title']}")

                    # --- Compliance verdict ---
                    st.markdown("---")
                    st.subheader("⚖️ Compliance Verdict")

                    compliance = data["compliance"]
                    status_val = compliance["status"]

                    if status_val == "compliant":
                        st.success("✅ COMPLIANT")
                    elif status_val == "non_compliant":
                        st.error("❌ NON-COMPLIANT")
                    else:
                        st.warning("⚠️ NEEDS REVIEW")

                    st.markdown(f"**Summary:** {compliance['summary']}")

                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.markdown("### 🚨 Issues Found")
                        if compliance["issues"]:
                            for issue in compliance["issues"]:
                                st.markdown(f"- {issue}")
                        else:
                            st.markdown("*No issues found.*")

                    with col_b:
                        st.markdown("### 💡 Recommendations")
                        if compliance["recommendations"]:
                            for rec in compliance["recommendations"]:
                                st.markdown(f"- {rec}")
                        else:
                            st.markdown("*No recommendations.*")

                    # Save video_id to session for use in query page
                    st.session_state["last_video_id"] = data["video_id"]
                    st.session_state["last_video_title"] = data["video_title"]

                    st.info(
                        f"💾 Video ID `{data['video_id']}` saved — "
                        "switch to the Query page to ask questions about it."
                    )

                else:
                    status.update(label="❌ Pipeline failed", state="error")
                    st.error(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.Timeout:
                status.update(label="❌ Request timed out", state="error")
                st.error(
                    "The request timed out after 5 minutes. "
                    "Try a shorter video (under 2 minutes recommended)."
                )
            except Exception as e:
                status.update(label="❌ Unexpected error", state="error")
                st.error(f"Unexpected error: {str(e)}")


# ---------------------------------------------------------------------------
# Page 2: Query Videos
# ---------------------------------------------------------------------------
elif page == "🔍 Query Videos":
    st.title("🔍 Query Indexed Videos")
    st.markdown(
        "Ask natural language compliance questions against all indexed videos, "
        "or scope your question to a specific video using its ID."
    )

    # Pre-fill video_id if coming from ingest page
    default_video_id = st.session_state.get("last_video_id", "")
    default_title = st.session_state.get("last_video_title", "")

    if default_video_id:
        st.info(f"💡 Last ingested: **{default_title}** (`{default_video_id}`)")

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Your Question",
            placeholder="What compliance issues were found in this video?",
        )
    with col2:
        video_id_filter = st.text_input(
            "Filter by Video ID (optional)",
            value=default_video_id,
            placeholder="Leave blank for all videos",
        )

    example_questions = [
        "What health claims were made?",
        "Are there any misleading guarantees?",
        "What disclaimers are missing?",
        "Is there any financial advice given?",
    ]
    st.markdown("**Example questions:**")
    cols = st.columns(len(example_questions))
    for i, eq in enumerate(example_questions):
        if cols[i].button(eq, key=f"eq_{i}"):
            question = eq

    if st.button("🔍 Search", disabled=not question):
        with st.spinner("Searching and generating answer..."):
            try:
                payload = {"question": question}
                if video_id_filter.strip():
                    payload["video_id"] = video_id_filter.strip()

                response = requests.post(
                    f"{API_BASE}/query",
                    json=payload,
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()

                    st.markdown("---")
                    st.subheader("🤖 AI Answer")
                    st.markdown(data["answer"])

                    st.markdown("---")
                    st.subheader(f"📚 Sources ({len(data['sources'])} chunks)")

                    for i, source in enumerate(data["sources"], 1):
                        source_icon = "🎙️" if source["source_type"] == "transcript" else "🔍"
                        with st.expander(
                            f"{source_icon} [{source['source_type'].upper()}] "
                            f"{source['video_title']} "
                            f"{'@ ' + str(source['start_time']) + 's' if source.get('start_time') else ''}",
                        ):
                            st.markdown(f"**Video ID:** `{source['video_id']}`")
                            st.markdown(f"**Text:**\n\n{source['text']}")

                elif response.status_code == 404:
                    st.warning(
                        "No relevant content found. "
                        "Make sure you've ingested at least one video first."
                    )
                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"Error: {str(e)}")