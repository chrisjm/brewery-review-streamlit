import spacy_streamlit
import streamlit as st

DEFAULT_TEXT = """During my San Diego visit, I hit this place up two times. Located in a house with a nice sized porch that has plenty of seats. The bar is just inside and it is full of old wood and is dark, and gives you a cool pub feel. Definitely a younger crowd here too. There were about 10-12 brews on tap, and they covered many styles including some sours. The Double IPA and high hopped pale ales were very good. Prices are reasonable and service was good. This is a must stop for anyone visiting the downtown area."""

spacy_model = "models/final"

st.title("Brewery Review NLP Analyzer")
text = st.text_area("Brewery review to analyze", DEFAULT_TEXT, height=200)
doc = spacy_streamlit.process_text(spacy_model, text)

spacy_streamlit.visualize_ner(
    doc,
    labels=["FEATURE", "BEER STYLE", "LOCATION"],
    show_table=True,
    title="Brewery features, beer styles, and location",
    sidebar_title="NER Labels",
)
st.text(f"Analyzed using spaCy model {spacy_model}")
