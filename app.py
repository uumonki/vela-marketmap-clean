import streamlit as st
from atlas import Atlas

model = Atlas('models/1m_60.bin', 'data/1m_with_topics.csv', 'data/topic_names.csv')
choices = st.selectbox("Function", ["Classify", "Keyword lookup", "Retrieve topic examples"])
if "Classify" in choices:
    desc = st.text_input("Company description:")
    if desc:
        i = model.classify_index(desc)
        st.write(str(i[0]) + " " + model.get_topic_name(int(i[0])))
if "Keyword lookup" in choices:
    keyword = st.text_input("Keyword:")
    if keyword:
        st.json(model.search_topics(keyword))
if "Retrieve topic examples" in choices:
    topic = st.number_input("Pick a topic index", 0, 196)
    st.write(model.get_topic_name(topic))
    st.dataframe(model.get_topic_companies(topic))
