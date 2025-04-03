from utils import transcription, update_session_state
import os
import streamlit as st
from transformers import HubertForCTC, Wav2Vec2Processor


def transcript_from_file(stt_tokenizer, stt_model):
    uploaded_file = st.file_uploader("", type=["mp3", "mp4", "wav"])

    if uploaded_file:
        st.success("✅ File uploaded successfully!")
        filename = uploaded_file.name
        transcription(stt_tokenizer, stt_model, filename, uploaded_file)


@st.cache_resource
def initialize_transcription_models():
    transcription_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    transcription_tokenizer = Wav2Vec2Processor.from_pretrained(
        "facebook/hubert-large-ls960-ft"
    )

    return transcription_tokenizer, transcription_model


def setup_page():
    st.set_page_config(
        page_title="Speech-to-Text Converter", page_icon="🎤", layout="wide"
    )

    # Ensure data directory exists
    # if not os.path.exists("../../data"):
    #     os.makedirs("../../data")

    # Custom CSS for improved UI
    st.markdown(
        """
        <style>
        .block-container { padding: 2rem; }
        .stRadio > label { font-weight: bold; }
        .stRadio > div { flex-direction: row; }
        p, span { text-align: justify; }
        .center-text { text-align: center; font-weight: bold; font-size: 18px; }
        .stFileUploader { text-align: center; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar for additional options
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("Customize your experience with available options below.")

    # Main UI
    st.title("🎤 Speech-to-Text Converter")
    st.info("📢 Upload your audio/video file, and we'll handle the transcription!")

    # Initialize session state variables
    if "page_index" not in st.session_state:
        st.session_state["audio_file"] = None
        st.session_state["process"] = []
        st.session_state["txt_transcript"] = ""
        st.session_state["page_index"] = 0

    st.divider()
    st.markdown(
        '<p class="center-text">🎙️ Let’s convert your speech to text effortlessly!</p>',
        unsafe_allow_html=True,
    )


def display_results():
    # Replay the uploaded audio
    if st.session_state.get("audio_file"):
        st.audio(st.session_state["audio_file"], format="audio/wav")

    # Button to reload a new file
    st.button(
        "🔄 Upload a New File",
        on_click=update_session_state,
        args=(
            "page_index",
            0,
        ),
    )

    st.divider()

    if st.session_state["process"] != []:
        with st.expander("🔍 Step-by-Step Transcription", expanded=True):
            for elt in st.session_state["process"]:
                # Timestamp
                st.write(elt[0])
                st.write(elt[1])

    st.divider()

    # Display final transcription text
    st.subheader("📝 Final Transcription")
    st.markdown(f"```{st.session_state['txt_transcript']}```")

    # Download transcription as a text file
    st.download_button(
        label="⬇️ Download Transcription",
        data=st.session_state["txt_transcript"],
        file_name="transcription.txt",
        mime="text/plain",
    )


def main():
    setup_page()

    tab1, tab2 = st.tabs(["🎥 Video Demonstration", "📜 Transcription Tool"])

    # First Tab - Video Upload & Playback
    with tab1:
        st.subheader("Watch the Video")
        video_path = "data/demo.mp4"
        st.video(video_path)  # Display the video
    with tab2:
        st.subheader("Transcription Method")

        # Default page - File Upload Section
        if st.session_state["page_index"] == 0:
            user_choice = st.radio("", ["Upload an audio/video file"])
            transcription_tokenizer, transcription_model = (
                initialize_transcription_models()
            )

            if user_choice == "Upload an audio/video file":
                transcript_from_file(transcription_tokenizer, transcription_model)

        # Results Page
        elif st.session_state["page_index"] == 1:
            display_results()


if __name__ == "__main__":
    main()
