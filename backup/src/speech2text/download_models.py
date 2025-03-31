"""
This script allows you to download the models used by our speech to text app from their respective librairies. These models are saved in a speech_to_text_app_models folder. This can save you some time when you initialize the app.
"""

from transformers import pipeline, Wav2Vec2Tokenizer, Wav2Vec2ForCTC, T5Tokenizer, T5ForConditionalGeneration, HubertForCTC, Wav2Vec2Processor
import pickle
import torch


def load_models():
    # 1 - English Speech to Text Model
    model_name = "facebook/hubert-large-ls960-ft"
    model = HubertForCTC.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    # Save the model and processor using torch.save
    torch.save(model.state_dict(), "models/STT_model_hubert-large-ls960-ft.pth")
    torch.save(processor, "models/STT_processor_hubert-large-ls960-ft.pth")

    # # 2 - Summarization Model
    # summarizer = pipeline("summarization")
    # pickle.dump(summarizer, open("models/summarizer.sav", 'wb'))

    # 3 - Other English Speech to Text Model
    model_name = "facebook/wav2vec2-large-960h-lv60-self"
    STT_tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
    STT_model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Save the model and tokenizer using torch.save
    torch.save(STT_model.state_dict(), "models/STT_model2_wav2vec2-large-960h-lv60-self.pth")
    torch.save(STT_tokenizer, "models/STT_tokenizer2_wav2vec2-large-960h-lv60-self.pth")

    # # 4 - Auto Punctuation Model
    # model_name = "flexudy/t5-small-wav2vec2-grammar-fixer"
    # T5_tokenizer = T5Tokenizer.from_pretrained(model_name)
    # T5_model = T5ForConditionalGeneration.from_pretrained(model_name)
    #
    # # (Here T5 Tokenizer can't be saved & loaded with pickle, as pickle serializes this model so it creates a local path in our
    # # object, which results in errors when app is not deployed locally anymore. We are going to use torch to save the model
    # torch.save(T5_tokenizer, "models/T5_tokenizer.sav")
    # torch.save(T5_model, "models/T5_model.sav")
    # # pickle.dump(T5_tokenizer, open("/workspace/speech_to_text_app_models/spiece.model", 'wb'))
    # # pickle.dump(T5_model, open("/workspace/speech_to_text_app_models/T5_model.sav", 'wb'))

    # 5 - Diarization model - Can't be saved anymore since pyannote.audio v2


if __name__ == '__main__':
    load_models()
    print("done")