import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import string
import re
from transformers import BartForConditionalGeneration, BartTokenizer



def summarizer(rawdocs):
    stopwords = list(STOP_WORDS)
    # print(stopwords)

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdocs)
    # print(doc)

    tokens = [token.text for token in doc]
    # print(tokens)

    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1
    # print(word_freq)

    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq
    # print(word_freq)

    sent_tokens = [sent for sent in doc.sents]
    # print(sent_tokens)

    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]
    # print(sent_tokens)

    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]
    # print(sent_scores)

    select_len = int(len(sent_tokens) * 0.3)
    # print(select_len)

    summary = nlargest(select_len, sent_scores, key=sent_scores.get)
    # print(summary)

    final_summary = [word.text for word in summary]
    summary = ''.join(final_summary)

    # print(text)
    # print(summary)
    # print("Length of original text ", len(text.split(' ')))
    # print("Length of summary text ", len(summary.split(' ')))

    return summary,doc,len(rawdocs.split(' ')),len(summary.split(' '))






#################################################################################


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
    # Load the BART model and tokenizer
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    # Additional preprocessing steps (you can customize these based on your needs)
    # Remove URLs and special characters
    text = remove_urls_and_special_chars(text)

    # Tokenize the input text
    inputs = bart_tokenizer("summarize: " + text, return_tensors="pt", max_length=max_length, truncation=True)

    # Generate the summary
    summary_ids = bart_model.generate(inputs["input_ids"], max_length=max_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary,text,len(text.split(' ')),len(summary.split(' '))





ans=generate_summary_bart_extended(text)
print(ans)

#########################################################################################################

