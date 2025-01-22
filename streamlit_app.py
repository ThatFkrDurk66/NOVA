import streamlit as st

# Set up the page
st.set_page_config(page_title="NOVA Assistant", layout="wide")  # Wide layout for better phone compatibility

# Add a compact title and description
st.title("NOVA Assistant")
st.markdown(
    """
    Welcome to NOVA! Your personal AI assistant.
    """
)

# Add a slider widget with compact layout
st.markdown("### Select a value:")
x = st.slider("", min_value=1, max_value=100, value=10, step=1)

# Display the result in a compact format
st.write(f"**You selected:** {x}")
st.write(f"**NOVA calculates:** {x}² = {x * x}")
