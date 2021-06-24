import pandas as pd
import streamlit as st
from annotated_text import annotated_text, annotation
from process_data import Ner, display_format

register = list()

nlp = Ner()
nlp.load_model()

desc = "Uses **Transformers** to check the sentences. By *Julio Cambronero Plaza*"

st.title('**Name Entity Recognition Model!**')
st.write(desc)

st.write('Sentence Input:')
sentence_input = st.text_input('Write the sentence here:')

if st.button('Compute text'):
    # register.append(sentence_input)
    # with open('registry.txt', 'a') as fp:
    #     fp.write('{}\n'.format(sentence_input))
    # print(sentence_input)

    df = nlp.get_prediction(sentence_input)
    text_tagged = display_format(df)
    annotated_text(*text_tagged)



