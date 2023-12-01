import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import string
import re
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
import time  # Import the time module

# Summarizer functions
def summarizer(rawdocs):
    stopwords = list(STOP_WORDS)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdocs)

    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1

    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq

    sent_tokens = [sent for sent in doc.sents]

    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]

    select_len = int(len(sent_tokens) * 0.3)

    summary = nlargest(select_len, sent_scores, key=sent_scores.get)
    final_summary = [word.text for word in summary]
    summary = ''.join(final_summary)

    return summary, doc, len(rawdocs.split(' ')), len(summary.split(' '))

def remove_non_printable(text):
    printable = set(string.printable)
    adjusted_text = ''.join(filter(lambda x: x in printable, text))
    return adjusted_text

def remove_urls_and_special_chars(text):
    adjusted_text = re.sub(r"http[s]?://\S+", "", text)
    adjusted_text = re.sub(r"http\S+\s+\(\d+\s+of\s+\d+\)\s+\[\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+\]", "", adjusted_text)
    adjusted_text = re.sub(r"http[s]?://freebooks\.by\.ru/view/CProgrammingLanguage/chapter1\.html", "", adjusted_text)
    adjusted_text = re.sub(r"\[\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+\]", "", adjusted_text)
    adjusted_text = re.sub(r"\(\d+\sof\s\d+\)\s\[\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+\]", "", adjusted_text)
    adjusted_text = remove_non_printable(adjusted_text)
    return adjusted_text

def generate_summary_bart_extended(text, max_length=1024):
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    text = remove_urls_and_special_chars(text)

    inputs = bart_tokenizer("summarize: " + text, return_tensors="pt", max_length=max_length, truncation=True)

    summary_ids = bart_model.generate(inputs["input_ids"], max_length=max_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary, text, len(text.split(' ')), len(summary.split(' '))

def generate_summary_t5_extended(text, model_name="t5-small", max_length=1024):
    # Load the T5 model and tokenizer
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
    t5_tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Additional preprocessing steps (you can customize these based on your needs)
    text = remove_urls_and_special_chars(text)

    # Tokenize the input text
    inputs = t5_tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the summary
    summary_ids = t5_model.generate(inputs["input_ids"], max_length=max_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary, text, len(text.split(' ')), len(summary.split(' '))

# Streamlit frontend code
st.set_page_config(page_title="SummarEase", page_icon="✨")  # Set page title and icon

# Project title and tagline
st.title("SummarEase 📄📃")
st.markdown("*Precision in Every Paragraph*")

# Input box for entering text in the sidebar
raw_text = st.sidebar.text_area("Enter your text here:")

# Display the number of words in the original text
st.sidebar.subheader("Original Text Statistics")
st.sidebar.write(f"Original Text Length: {len(raw_text.split(' '))} words")

# Dropdown menu for selecting summarizer
summarizer_name = st.sidebar.selectbox("Select Summarizer:", ["Spacy", "BART", "T5 Model"])

# Increase the width of the left sidebar to 40%
st.markdown(
    """
    <style>
        .css-145kmo2 {
            width: 40%;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a spinner for loading animation
with st.spinner(f"Generating {summarizer_name} Summary..."):
    # Sleep for a short duration to simulate processing
    time.sleep(2)

    # Perform summarization based on the selected summarizer
    if st.sidebar.button("Generate Summary"):
        if summarizer_name == "Spacy":
            summary, _, _, _ = summarizer(raw_text)
        elif summarizer_name == "BART":
            summary, _, _, _ = generate_summary_bart_extended(raw_text)
        elif summarizer_name == "T5 Model":
            summary, _, _, _ = generate_summary_t5_extended(raw_text)

        # Display the results in the main window
        st.subheader(f"{summarizer_name} Summary")
        st.write(summary)

        st.subheader("Summary Statistics")
        st.write(f"{summarizer_name} Summary Length: {len(summary.split(' '))} words")
