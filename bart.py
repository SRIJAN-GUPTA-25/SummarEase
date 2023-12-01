import re
import string
from transformers import BartForConditionalGeneration, BartTokenizer

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

def remove_specific_url_from_summary(summary):
    specific_url_pattern = r"http[s]?://freebooks\.by\.ru/view/CProgrammingLanguage/chapter1\.html"
    summary_without_url = re.sub(specific_url_pattern, "", summary)
    summary_without_url = re.sub(r"\[\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+\]", "", summary_without_url)
    summary_without_url = re.sub(r"\(\d+\sof\s\d+\)\s\[\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+\]", "", summary_without_url)
    return summary_without_url

def generate_summary_bart(text, model, tokenizer, max_length=2000):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load BART model and tokenizer
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Provide the input text
input_text = """Samsung recently cancelled its in-person MWC 2021 event, instead, committing to an online-only format. The South Korean tech giant
recently made it official, setting a time and date for the Samsung Galaxy MWC Virtual Event.
The event will be held on June 28 at 17:15 UTC (22:45 IST) and will be live-streamed on YouTube. In its release, Samsung says that it will
introduce its "ever-expanding Galaxy device ecosystem". Samsung also plans to present the latest technologies and innovation efforts in
relation to the growing importance of smart device security.
Samsung will also showcase its vision for the future of smartwatches to provide new experiences for users and new opportunities for
developers. Samsung also shared an image for the event with silhouettes of a smartwatch, a smartphone, a tablet and a laptop."""

# Preprocess the input text
input_text_cleaned = remove_urls_and_special_chars(input_text)

# Generate the summary using BART model
summary = generate_summary_bart(input_text_cleaned, bart_model, bart_tokenizer, max_length=2000)  # Adjust max_length as needed
summary = remove_specific_url_from_summary(summary)

# Display the generated summary
print(" --------------------------------------------  ")
print(f"Generated Summary:")
print(" --------------------------------------------  ")
print(f" {summary}")
print(" --------------------------------------------  ")
