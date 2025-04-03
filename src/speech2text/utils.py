import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Tokenizer
import audioread
import librosa
from pydub import AudioSegment, silence
from datetime import timedelta
import os
import streamlit as st
import time


def get_middle_silence_time(silence_list):
    length = len(silence_list)
    index = 0
    while index < length:
        diff = silence_list[index][1] - silence_list[index][0]
        if diff < 3500:
            silence_list[index] = silence_list[index][0] + diff / 2
            index += 1
        else:
            adapted_diff = 1500
            silence_list.insert(index + 1, silence_list[index][1] - adapted_diff)
            silence_list[index] = silence_list[index][0] + adapted_diff

            length += 1
            index += 2

    return silence_list


def clean_directory(path):
    for file in os.listdir(path):
        os.remove(os.path.join(path, file))


def silences_distribution(
    silence_list, min_space, max_space, start, end, srt_token=False
):
    # If starts != 0, we need to adjust end value since silences detection is performed on the trimmed/cut audio
    # (and not on the original audio) (ex: trim audio from 20s to 2m will be 0s to 1m40 = 2m-20s)

    # Shift the end according to the start value
    end -= start
    start = 0
    end *= 1000

    # Step 1 - Add start value
    newsilence = [start]

    # Step 2 - Create a regular distribution between start and the first element of silence_list to don't have a gap > max_space and run out of memory
    # example newsilence = [0] and silence_list starts with 100000 => It will create a massive gap [0, 100000]

    if silence_list[0] - max_space > newsilence[0]:
        for i in range(
            int(newsilence[0]), int(silence_list[0]), max_space
        ):  # int bc float can't be in a range loop
            value = i + max_space
            if value < silence_list[0]:
                newsilence.append(value)

    # Step 3 - Create a regular distribution until the last value of the silence_list
    min_desired_value = newsilence[-1]
    max_desired_value = newsilence[-1]
    nb_values = len(silence_list)

    while nb_values != 0:
        max_desired_value += max_space

        # Get a window of the values greater than min_desired_value and lower than max_desired_value
        silence_window = list(
            filter(lambda x: min_desired_value < x <= max_desired_value, silence_list)
        )

        if silence_window != []:
            # Get the nearest value we can to min_desired_value or max_desired_value depending on srt_token
            if srt_token:
                nearest_value = min(
                    silence_window, key=lambda x: abs(x - min_desired_value)
                )
                nb_values -= (
                    silence_window.index(nearest_value) + 1
                )  # (index begins at 0, so we add 1)
            else:
                nearest_value = min(
                    silence_window, key=lambda x: abs(x - max_desired_value)
                )
                # Max value index = len of the list
                nb_values -= len(silence_window)

            # Append the nearest value to our list
            newsilence.append(nearest_value)

        # If silence_window is empty we add the max_space value to the last one to create an automatic cut and avoid multiple audio cutting
        else:
            newsilence.append(newsilence[-1] + max_space)

        min_desired_value = newsilence[-1]
        max_desired_value = newsilence[-1]

    # Step 4 - Add the final value (end)

    if end - newsilence[-1] > min_space:
        # Gap > Min Space
        if end - newsilence[-1] < max_space:
            newsilence.append(end)
        else:
            # Gap too important between the last list value and the end value
            # We need to create automatic max_space cut till the end
            newsilence = generate_regular_split_till_end(
                newsilence, end, min_space, max_space
            )
    else:
        # Gap < Min Space <=> Final value and last value of new silence are too close, need to merge
        if len(newsilence) >= 2:
            if end - newsilence[-2] <= max_space:
                # Replace if gap is not too important
                newsilence[-1] = end
            else:
                newsilence.append(end)

        else:
            if end - newsilence[-1] <= max_space:
                # Replace if gap is not too important
                newsilence[-1] = end
            else:
                newsilence.append(end)

    return newsilence


def generate_regular_split_till_end(time_list, end, min_space, max_space):
    # In range loop can't handle float values so we convert to int
    int_last_value = int(time_list[-1])
    int_end = int(end)

    # Add maxspace to the last list value and add this value to the list
    for i in range(int_last_value, int_end, max_space):
        value = i + max_space
        if value < end:
            time_list.append(value)

    # Fix last automatic cut
    # If small gap (ex: 395 000, with end = 400 000)
    if end - time_list[-1] < min_space:
        time_list[-1] = end
    else:
        # If important gap (ex: 311 000 then 356 000, with end = 400 000, can't replace and then have 311k to 400k)
        time_list.append(end)
    return time_list


def init_transcription(start, end):
    st.write("Transcription between", start, "and", end, "seconds in process.\n\n")
    txt_text = ""
    srt_text = ""
    save_result = []
    return txt_text, srt_text, save_result


def detect_silences(audio):
    # Get Decibels (dB) so silences detection depends on the audio instead of a fixed value
    dbfs = audio.dBFS

    # Get silences timestamps > 750ms
    silence_list = silence.detect_silence(
        audio, min_silence_len=750, silence_thresh=dbfs - 14
    )

    return silence_list


def transcribe_audio_part(
    filename, stt_model, stt_tokenizer, myaudio, sub_start, sub_end, index
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        with torch.no_grad():
            new_audio = myaudio[sub_start:sub_end]  # Works in milliseconds
            path = "audio_" + str(index) + ".mp3"
            new_audio.export(path)  # Exports to a mp3 file in the current path

            # Load audio file with librosa, set sound rate to 16000 Hz because the model we use was trained on 16000 Hz data
            input_audio, _ = librosa.load(path, sr=16000)

            # return PyTorch torch.Tensor instead of a list of python integers thanks to return_tensors = ‘pt’
            input_values = (
                stt_tokenizer(input_audio, return_tensors="pt").to(device).input_values
            )

            # Get logits from the data structure containing all the information returned by the model and get our prediction
            logits = stt_model.to(device)(input_values).logits
            prediction = torch.argmax(logits, dim=-1)

            # Decode & lower our string (model's output is only uppercase)
            if isinstance(stt_tokenizer, Wav2Vec2Tokenizer):
                transcription = stt_tokenizer.batch_decode(prediction)[0]
            elif isinstance(stt_tokenizer, Wav2Vec2Processor):
                transcription = stt_tokenizer.decode(prediction[0])

            # return transcription
            return transcription.lower()

    except audioread.NoBackendError:
        # Means we have a chunk with a [value1 : value2] case with value1>value2
        st.error(
            "Sorry, seems we have a problem on our side. Please change start & end values."
        )
        time.sleep(3)
        st.stop()


def display_transcription(
    transcription, save_result, txt_text, srt_text, sub_start, sub_end
):
    temp_timestamps = (
        str(timedelta(milliseconds=sub_start)).split(".")[0]
        + " --> "
        + str(timedelta(milliseconds=sub_end)).split(".")[0]
        + "\n"
    )
    temp_list = [temp_timestamps, transcription, int(sub_start / 1000)]
    save_result.append(temp_list)
    st.write(temp_timestamps)
    st.write(transcription + "\n\n")
    txt_text += transcription + " "  # So x seconds sentences are separated

    return save_result, txt_text, srt_text


def transcription_non_diarization(
    filename,
    myaudio,
    start,
    end,
    srt_token,
    stt_model,
    stt_tokenizer,
    min_space,
    max_space,
    save_result,
    txt_text,
    srt_text,
):
    # get silences
    silence_list = detect_silences(myaudio)
    if silence_list != []:
        silence_list = get_middle_silence_time(silence_list)
        silence_list = silences_distribution(
            silence_list, min_space, max_space, start, end, srt_token
        )
    else:
        silence_list = generate_regular_split_till_end(
            silence_list, int(end), min_space, max_space
        )

    # Transcribe each audio chunk (from timestamp to timestamp) and display transcript
    for i in range(0, len(silence_list) - 1):
        sub_start = silence_list[i]
        sub_end = silence_list[i + 1]

        transcription = transcribe_audio_part(
            filename, stt_model, stt_tokenizer, myaudio, sub_start, sub_end, i
        )

        if transcription != "":
            save_result, txt_text, srt_text = display_transcription(
                transcription, save_result, txt_text, srt_text, sub_start, sub_end
            )

    return save_result, txt_text, srt_text


def update_session_state(var, data, concatenate_token=False):
    """
    A simple function to update a session state variable
    :param var: variable's name
    :param data: new value of the variable
    :param concatenate_token: do we replace or concatenate
    """

    if concatenate_token:
        st.session_state[var] += data
    else:
        st.session_state[var] = data


def transcription(stt_tokenizer, stt_model, filename, uploaded_file=None):
    # Handle case where uploaded_file is None (e.g., from YouTube extraction)
    if uploaded_file is None:
        uploaded_file = filename

    # Get audio length
    myaudio = AudioSegment.from_file(uploaded_file)
    audio_length = myaudio.duration_seconds

    # Display audio preview
    st.subheader("🎵 Preview Your Audio")
    st.audio(uploaded_file)
    update_session_state("audio_file", uploaded_file)

    st.divider()

    # Check if transcription is possible
    if audio_length > 0:
        st.info(f"📏 Audio Duration: **{round(audio_length, 2)} seconds**")

        # Button to start transcription
        transcript_btn = st.button("🎙️ Start Transcription")

        if transcript_btn:
            # Show progress spinner
            with st.spinner("⏳ Transcribing your audio... Please wait."):
                start, end = 0, int(audio_length)
                txt_text, srt_text, save_result = init_transcription(start, end)
                srt_token = False
                min_space, max_space = 25000, 45000

                # Perform transcription (Non-Diarization Mode)
                save_result, txt_text, srt_text = transcription_non_diarization(
                    f"../data/{filename}",
                    myaudio,
                    start,
                    end,
                    srt_token,
                    stt_model,
                    stt_tokenizer,
                    min_space,
                    max_space,
                    save_result,
                    txt_text,
                    srt_text,
                )
                update_session_state("process", save_result)

                # Clean up temp files
                # clean_directory("../../data")

                # Display results
                if txt_text:
                    update_session_state("txt_transcript", txt_text)
                    st.success("✅ Transcription completed successfully!")

                    st.subheader("📝 Transcribed Text")
                    st.markdown(f"```{txt_text}```")

                    # Download button
                    st.download_button(
                        label="⬇️ Download Transcription",
                        data=txt_text,
                        file_name="my_transcription.txt",
                        mime="text/plain",
                        on_click=update_session_state,
                        args=(
                            "page_index",
                            1,
                        ),
                    )
                else:
                    st.error(
                        "⚠️ Transcription failed. Please check your audio file or parameters."
                    )

    else:
        st.error(
            "🚨 Your audio file seems to have **0 seconds duration**. Please upload a valid file."
        )
        time.sleep(3)
        st.stop()
