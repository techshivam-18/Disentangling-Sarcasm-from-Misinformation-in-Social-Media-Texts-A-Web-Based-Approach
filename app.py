import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt # Import matplotlib for plotting
import json # Needed for JSON serialization/deserialization for feedback.py

# Streamlit UI configuration MUST be the very first Streamlit command
st.set_page_config(page_title="Sarcasm & Misinformation Detector", layout="wide")

# Import functions from your utility modules
from util.models_utils import load_models, predict_text, predict_file
from util.reddit_utils import fetch_reddit_posts
from util.auth import login, logout, signup, is_logged_in, get_current_user # Now uses SQLite auth
from util.feedback import save_feedback, save_nlp_feedback # Assuming this is updated for SQLite

from util.nlp_utils import (
    preprocess_text, pos_tags, get_sentiment,
    get_embedding, topic_modeling, show_wordcloud
)

# Load models
mis_model, mis_vectorizer, sar_model, sar_vectorizer = load_models()

st.title("🧠 Disentangling Sarcasm from Misinformation in Social Media Texts")

# Authentication Section
if not is_logged_in():
    auth_action = st.sidebar.radio("Login / Signup", ["Login", "Signup"])
    if auth_action == "Login":
        login() # Calls the SQLite-based login
    else:
        signup() # Calls the SQLite-based signup
else:
    st.sidebar.success(f"Welcome, {get_current_user()}!")
    if st.sidebar.button("Logout"):
        logout() # Calls the SQLite-based logout
        st.rerun()

    # Sidebar navigation for logged-in users
    option = st.sidebar.radio("Go to", [
        "🏠 Home", "📄 Detect Text", "🌐 Reddit Live", "📝 Feedback", "🧠 NLP Toolkit"
    ])

    if option == "🏠 Home":
        st.markdown("""
            <div style="text-align:center; padding-top:20px;">
                <img src="https://cdn.pixabay.com/photo/2020/06/12/23/50/coding-5288078_1280.png" width="80%" style="border-radius: 10px;">
                <h1 style="margin-top:30px; font-size: 36px; background: -webkit-linear-gradient(#0077b6, #00b4d8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    Welcome to Sarcasm & Misinformation Detector
                </h1>
                <p style="font-size:18px; color:#212529; max-width: 700px; margin:auto;">
                    Discover whether a statement is sarcastic or misleading using <strong>AI-powered XGBoost models</strong>. Dive into Reddit in real time and explore what’s real and what’s just ironic!
                </p>
                <hr style="margin-top:40px; margin-bottom:30px; border-color:#00b4d8;">
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="display:flex; flex-direction: column; gap: 20px;">
                <div style="background-color: #d0f0fd; padding: 20px; border-radius: 10px; border-left: 5px solid #00b4d8; color: #003459;">
                    <h4>🧠 Powerful Detection</h4>
                    <p>Uses advanced XGBoost models to detect both sarcasm and misinformation in user-generated content.</p>
                </div>
                <div style="background-color: #caf0f8; padding: 20px; border-radius: 10px; border-left: 5px solid #0096c7; color: #003459;">
                    <h4>🌐 Reddit Integration</h4>
                    <p>Fetch real-time posts from Reddit and analyze them instantly for sarcasm and misinformation.</p>
                </div>
                <div style="background-color: #ade8f4; padding: 20px; border-radius: 10px; border-left: 5px solid #0077b6; color: #003459;">
                    <h4>💬 Feedback Driven</h4>
                    <p>We value your input! Use the feedback form to help us improve the platform for everyone.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    elif option == "📄 Detect Text":
        st.subheader("📄 Text Input Detection")
        input_text = st.text_area("Enter your text")

        col1, col2 = st.columns(2)
        if col1.button("Detect Sarcasm"):
            _, sar_result = predict_text(input_text, mis_model, mis_vectorizer, sar_model, sar_vectorizer)
            st.success(f"Sarcasm Detection Result: {sar_result}")
            # Graph for single text sarcasm
            st.markdown("### 📊 Sarcasm Prediction")
            prediction_label = "Sarcastic" if sar_result == 1 else "Not Sarcastic"
            sar_data = pd.DataFrame({'Prediction': [prediction_label], 'Count': [1]})
            st.bar_chart(sar_data.set_index('Prediction'))

        if col2.button("Detect Misinformation"):
            mis_result, _ = predict_text(input_text, mis_model, mis_vectorizer, sar_model, sar_vectorizer)
            st.success(f"Misinformation Detection Result: {mis_result}")
            # Graph for single text misinformation
            st.markdown("### 📊 Misinformation Prediction")
            prediction_label = "Misinformation" if mis_result == 1 else "Not Misinformation"
            mis_data = pd.DataFrame({'Prediction': [prediction_label], 'Count': [1]})
            st.bar_chart(mis_data.set_index('Prediction'))

        st.markdown("---")
        st.subheader("📁 Upload CSV/Text File")
        uploaded_file = st.file_uploader("Upload file", type=["csv", "txt"])

        if uploaded_file is not None:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    st.write("Analyzing 'text' column from CSV...")
                    df['misinformation_prediction'], df['sarcasm_prediction'] = zip(*df['text'].apply(
                        lambda x: predict_text(x, mis_model, mis_vectorizer, sar_model, sar_vectorizer)
                    ))
                    st.dataframe(df)

                    # Graphs for CSV analysis
                    st.markdown("### 📊 CSV Analysis Results")
                    
                    # Misinformation Bar Chart
                    mis_counts = df['misinformation_prediction'].value_counts().reset_index()
                    mis_counts.columns = ['Prediction', 'Count']
                    mis_counts['Prediction'] = mis_counts['Prediction'].map({1: 'Misinformation', 0: 'Not Misinformation'})
                    st.subheader("Misinformation Distribution")
                    st.bar_chart(mis_counts.set_index('Prediction'))

                    # Sarcasm Bar Chart
                    sar_counts = df['sarcasm_prediction'].value_counts().reset_index()
                    sar_counts.columns = ['Prediction', 'Count']
                    sar_counts['Prediction'] = sar_counts['Prediction'].map({1: 'Sarcastic', 0: 'Not Sarcastic'})
                    st.subheader("Sarcasm Distribution")
                    st.bar_chart(sar_counts.set_index('Prediction'))

                else:
                    st.error("CSV must contain a 'text' column for analysis.")
            elif uploaded_file.type == "text/plain":
                file_content = uploaded_file.read().decode("utf-8")
                mis_pred_file, sar_pred_file = predict_file(file_content, mis_model, mis_vectorizer, sar_model, sar_vectorizer)
                st.write(f"File Misinformation Prediction: {mis_pred_file}")
                st.write(f"File Sarcasm Prediction: {sar_pred_file}")
                # Graph for single text from file
                st.markdown("### 📊 File Predictions")
                file_predictions = pd.DataFrame({
                    'Type': ['Misinformation', 'Sarcasm'],
                    'Prediction': [
                        "Misinformation" if mis_pred_file == 1 else "Not Misinformation",
                        "Sarcastic" if sar_pred_file == 1 else "Not Sarcastic"
                    ]
                })
                st.dataframe(file_predictions) # Display as table for clarity of two predictions
                # You could also use a bar chart if you convert predictions to counts (e.g., 1 for detected, 0 for not)
                # For example, if you want to count how many types of 'detection' occurred:
                # detected_types = []
                # if mis_pred_file == 1: detected_types.append('Misinformation')
                # if sar_pred_file == 1: detected_types.append('Sarcasm')
                # if detected_types:
                #     detected_df = pd.DataFrame({'Detection Type': detected_types, 'Count': [1]*len(detected_types)})
                #     st.bar_chart(detected_df.set_index('Detection Type'))


            else:
                st.warning("Unsupported file type. Please upload a CSV or TXT file.")


    elif option == "🌐 Reddit Live":
        st.subheader("🌐 Real-time Reddit Analysis")
        search_query = st.text_input("Enter Search Query (e.g., 'news', 'politics')", value="worldnews")
        num_posts = st.slider("Number of Posts", min_value=1, max_value=50, value=10)

        if st.button("Fetch & Analyze"):
            posts_df = fetch_reddit_posts(search_query, num_posts)
            if not posts_df.empty:
                posts_df["misinformation_prediction"], posts_df["sarcasm_prediction"] = zip(*posts_df["title"].apply(
                    lambda x: predict_text(x, mis_model, mis_vectorizer, sar_model, sar_vectorizer)
                ))
                st.dataframe(posts_df[["title", "misinformation_prediction", "sarcasm_prediction"]])

                # Graphs for Reddit Live analysis
                st.markdown("### 📊 Reddit Post Analysis Results")
                
                # Misinformation Distribution
                mis_counts_reddit = posts_df['misinformation_prediction'].value_counts().reset_index()
                mis_counts_reddit.columns = ['Prediction', 'Count']
                mis_counts_reddit['Prediction'] = mis_counts_reddit['Prediction'].map({1: 'Misinformation', 0: 'Not Misinformation'})
                st.subheader("Misinformation Distribution (Reddit)")
                st.bar_chart(mis_counts_reddit.set_index('Prediction'))

                # Sarcasm Distribution
                sar_counts_reddit = posts_df['sarcasm_prediction'].value_counts().reset_index()
                sar_counts_reddit.columns = ['Prediction', 'Count']
                sar_counts_reddit['Prediction'] = sar_counts_reddit['Prediction'].map({1: 'Sarcastic', 0: 'Not Sarcastic'})
                st.subheader("Sarcasm Distribution (Reddit)")
                st.bar_chart(sar_counts_reddit.set_index('Prediction'))

            else:
                st.warning("No posts found or error fetching data.")

    elif option == "📝 Feedback":
        st.subheader("📝 Submit Feedback")
        user = get_current_user()
        feedback_text = st.text_area("Write your feedback here")
        if st.button("Submit Feedback"):
            if feedback_text.strip():
                save_feedback(user, feedback_text)
                st.success("✅ Feedback submitted!")
            else:
                st.warning("Please enter feedback before submitting.")

    elif option == "🧠 NLP Toolkit":
        user = get_current_user()

        st.sidebar.header("🧠 NLP Toolkit")
        if st.sidebar.checkbox("Enable NLP Analysis"):
            st.subheader("🧠 Natural Language Processing")
            input_text = st.text_area("Enter text for NLP analysis")

            if st.button("Run NLP Analysis") and input_text.strip():
                st.markdown("### 🔹 Preprocessed Text")
                cleaned = preprocess_text(input_text)
                st.code(cleaned)

                st.markdown("### 🔹 POS Tags")
                tags = pos_tags(input_text)
                st.write(tags)

                st.markdown("### 🔹 Sentiment Analysis")
                sentiment = get_sentiment(input_text)
                st.json(sentiment) # Display raw sentiment values for debugging

                # --- IMPROVED SENTIMENT VISUALIZATION ---
                st.markdown("### 📊 Sentiment Visualization")
                
                # Display sentiment as metrics for direct values
                col_pol, col_sub = st.columns(2)
                with col_pol:
                    st.metric(label="Polarity (Emotion)", value=f"{sentiment['polarity']:.2f}")
                with col_sub:
                    st.metric(label="Subjectivity (Opinion)", value=f"{sentiment['subjectivity']:.2f}")

                # Create a Matplotlib figure for the bar chart for better control
                fig, ax = plt.subplots(figsize=(7, 4))
                metrics = ['Polarity', 'Subjectivity']
                values = [sentiment['polarity'], sentiment['subjectivity']]
                colors = ['skyblue' if v >= 0 else 'salmon' for v in values] # Color based on polarity

                ax.bar(metrics, values, color=colors)
                ax.set_ylabel('Score')
                ax.set_title('Sentiment Scores')
                
                # Set y-axis limits to provide context, especially for polarity
                ax.set_ylim([-1.0, 1.0]) # Polarity ranges from -1 to +1
                
                # Add a horizontal line at y=0 for polarity reference
                ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')

                st.pyplot(fig)
                plt.close(fig) # Close the figure to free memory

                # --- END IMPROVED SENTIMENT VISUALIZATION ---


                st.markdown("### 🔹 Word Embedding (vector shape)")
                embedding = get_embedding(input_text)
                st.write(f"Shape: {embedding.shape}")
                st.write(embedding[:10])

                st.markdown("### 🔹 Topic Modeling")
                topics = topic_modeling([input_text])
                st.write(topics)

                st.markdown("### 🔹 Word Cloud")
                show_wordcloud(input_text)

                save_nlp_feedback(user, input_text, cleaned, tags, sentiment, topics)
