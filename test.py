import streamlit as st

# Set the title of the app
st.title("My First Streamlit App")

# Add some text
st.write("Hello, world! This is a simple boilerplate for a Streamlit app.")

# Add a button
st.subheader("Try the Model")
st.markdown(
            "Paste text, pick a model/feature/k, and see the predicted topic with scores."
        )
with st.popover("Sample Text Links"):
            st.header("Sci.space")
            st.markdown("https://www.nasa.gov/missions/")
            st.markdown("https://en.wikipedia.org/wiki/Space_exploration")

            st.header("Sci.med")
            st.markdown("https://www.who.int/news-room/fact-sheets")
            st.markdown("https://en.wikipedia.org/wiki/Medical_diagnosis")

            st.header("Rec.sport")
            st.markdown("https://www.fifa.com/tournaments/mens/worldcup")
            st.markdown("https://en.wikipedia.org/wiki/Association_football")

            st.header("Rec.autos")

            st.markdown("https://www.motortrend.com/cars/")
            st.markdown("https://en.wikipedia.org/wiki/Automobile_engine")


            st.header("Talk.politics")

            st.markdown("https://www.nytimes.com/international/section/politics")
            st.markdown("https://en.wikipedia.org/wiki/Political_ideology")
        