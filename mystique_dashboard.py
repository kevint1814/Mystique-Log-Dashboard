# Streamlit Configuration - must be the first command in the script
import streamlit as st
st.set_page_config(page_title="Mystique Dashboard", layout="wide")

# Now you can import other libraries
import pandas as pd
import os
from sklearn.ensemble import IsolationForest
import altair as alt
import requests
import plotly.express as px
from streamlit_lottie import st_lottie
from langdetect import detect
from googletrans import Translator

translator = Translator()

# Function to load Lottie animations
def load_lottie_url(url:str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animation for the homepage
lottie_animation = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_uyzvm6uu.json")  # Lottie URL
if lottie_animation:
    st_lottie(lottie_animation, speed=1, width=700, height=400, key="home_animation")

# Load the dataset
file_path = "Clear_Anomalies_Telecom_Logs_Dataset.csv."
telecom_logs = pd.read_csv(file_path)

# GroqCloud API Key
API_KEY = os.getenv("GROQCLOUD_API_KEY")


# Anomaly detection function
def detect_anomalies(logs, sensitivity=0.3):
    log_level_mapping = {"INFO": 1, "WARNING": 2, "ERROR": 3}
    logs["log_level_numeric"] = logs["log_level"].map(log_level_mapping)
    logs["message_length"] = logs["message"].str.len()
    model = IsolationForest(contamination=sensitivity, random_state=42)
    logs["anomaly_prediction"] = model.fit_predict(logs[["log_level_numeric", "message_length"]])
    logs["anomaly"] = logs["anomaly_prediction"].apply(lambda x: "Anomaly" if x == -1 else "Normal")
    return logs

# Enhanced AI explanation function
def get_ai_explanation(prompt, api_key, max_tokens=500, retries=3):
    """
    Fetch AI-generated explanation with retries and error handling.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    for attempt in range(retries):
        payload = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                return content  # Return valid explanation
        else:
            st.error(f"Error {response.status_code}: {response.json().get('error', {}).get('message', 'Unknown error')}")
    
    return "Explanation could not be retrieved. Please try again."

# AI-Powered log summarization
def summarize_logs_with_ai(api_key, logs, num_logs=5):
    formatted_logs = "\n".join([
        f"At {row['timestamp']}, {row['device']} experienced: {row['message']} ({row['log_level']} level)."
        for _, row in logs.head(num_logs).iterrows()
    ])
    prompt = f"Summarize the following logs:\n{formatted_logs}"
    return get_ai_explanation(prompt, api_key)

# Root cause analysis based on logs
def root_cause_analysis(logs):
    causes = []
    for _, row in logs.iterrows():
        if "connection timeout" in row["message"].lower():
            causes.append([row['timestamp'], row['device'], "Likely router failure due to connection timeout."])
        elif "device unreachable" in row["message"].lower():
            causes.append([row['timestamp'], row['device'], "Might be experiencing network connectivity issues."])
        elif "device overheating" in row["message"].lower():
            causes.append([row['timestamp'], row['device'], "Might have hardware overheating problems."])
        elif "authentication failure" in row["message"].lower():
            causes.append([row['timestamp'], row['device'], "Might be experiencing user authentication issues."])
        elif "high latency" in row["message"].lower():
            causes.append([row['timestamp'], row['device'], "Might be overloaded, causing high latency."])
        else:
            causes.append([row['timestamp'], row['device'], f"Unclassified issue: {row['message']}"])
    return causes

# Chatbot with predefined responses
def translate_query(query, target_language="en"):
    detected_language = detect(query)
    if detected_language != target_language:
        translated_text = translator.translate(query, src=detected_language, dest=target_language).text
        return translated_text, detected_language
    return query, target_language

def get_enhanced_chatbot_response(user_input, api_key):
    predefined_responses = {
        "What does this app do?": "This app analyzes telecom logs using AI. It detects anomalies, summarizes logs, and helps identify root causes of network outages.",
        "How does anomaly detection work?": "Anomaly detection identifies unusual patterns in the logs that might indicate potential issues in the telecom network.",
        "What is root cause analysis?": "Root cause analysis identifies the underlying cause of network issues by analyzing logs and anomalies, helping operators resolve problems faster.",
        "How does the visualization feature work?": "The visualization feature provides insights into the log data through bar charts, pie charts, time series plots, and heatmaps.",
        "Can I filter logs by device?": "Yes, in the 'Root Cause Analysis' section, you can filter logs by device using the interactive filter.",
        "What is the purpose of the chatbot?": "The chatbot helps answer questions about the logs, anomaly detection, root cause analysis, and the app's features.",
        "How can I download the anomaly data?": "In the 'Anomaly Detection' section, after detecting anomalies, you can use the download button to save the anomaly data as a CSV file.",
        "What does the heatmap show?": "The heatmap visualizes the distribution of log levels (INFO, WARNING, ERROR) across devices, helping identify patterns or anomalies.",
        "How are anomalies detected?": "Anomalies are detected using machine learning models, such as Isolation Forest, which analyze log attributes to identify unusual patterns.",
        "Can the app summarize logs?": "Yes, the app uses AI to provide summaries of the logs in simple terms, making it easier to understand key events.",
        "What is the most common log level?": "The most common log level depends on the dataset. Use the 'Visualizations' section to analyze log-level distributions.",
        "What is the 'sensitivity' slider in anomaly detection?": "The sensitivity slider adjusts the threshold for anomaly detection. Higher sensitivity detects more anomalies but may include false positives.",
        "How do I interpret the time series plot?": "The time series plot shows trends in log levels over time, highlighting spikes or drops in log activity.",
        "How do I analyze logs for a specific device?": "You can use the interactive filter in the 'Root Cause Analysis' section to focus on logs from a specific device.",
        "What kind of logs does this app process?": "The app processes telecom network logs, including timestamps, devices, log levels (INFO, WARNING, ERROR), and messages.",
        "What should I do if the app doesn't detect anomalies?": "If no anomalies are detected, try adjusting the sensitivity slider in the 'Anomaly Detection' section or ensure the dataset is properly loaded.",
        "How does AI explain visualizations?": "The app uses AI to generate explanations for visualizations, providing insights into the patterns and trends shown in the charts.",
        "What are common root causes in telecom logs?": "Common root causes include connection timeouts, device unreachability, authentication failures, high latency, and device overheating.",
        "Can I provide feedback on the chatbot?": "Yes, you can use the feedback section in the chatbot to indicate if the response was helpful."
    }

    # Dynamic context awareness
    if user_input in predefined_responses:
        return predefined_responses[user_input]

    prompt = user_input
    return get_ai_explanation(prompt, api_key, max_tokens=300)

# Streamlit UI
# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Home", "Anomaly Detection", "Root Cause Analysis", "Visualizations", "Chatbot", "Conclusion"])

if option == "Home":
    st.title("Mystique: Telecom Network Log Analysis Dashboard")
    st.markdown("""
        Welcome to **Mystique**, your AI-powered tool for analyzing telecom network logs.
        - 📊 Detect anomalies
        - 🤖 Summarize logs with AI
        - 🛠 Identify root causes of outages
    """)

elif option == "Anomaly Detection":
    st.title("Anomaly Detection")
    sensitivity = st.slider("Set anomaly detection sensitivity", 0.01, 0.5, 0.3)
    processed_logs = detect_anomalies(telecom_logs, sensitivity)
    anomalies = processed_logs[processed_logs["anomaly"] == "Anomaly"]
    st.subheader("Detected Anomalies")
    if not anomalies.empty:
        st.dataframe(anomalies)
        csv = anomalies.to_csv(index=False)
        st.download_button("Download Anomalies", csv, "anomalies.csv")

        # AI Explanation for detected anomalies
        total_logs = len(telecom_logs)
        anomaly_count = len(anomalies)
        prompt = (
            f"We detected {anomaly_count} anomalies out of {total_logs} logs in a telecom network. "
            "Explain the significance of these anomalies in simple terms for a non-technical audience."
        )
        explanation = get_ai_explanation(prompt, API_KEY, max_tokens=400)
        st.markdown(f"**AI Explanation:** {explanation}")
    else:
        st.warning("No anomalies detected with the current sensitivity.")

elif option == "Root Cause Analysis":
    st.title("Root Cause Analysis")
    st.markdown("""
        This section identifies potential root causes of issues in the telecom network based on log messages.
    """)
    root_causes = root_cause_analysis(telecom_logs)
    if root_causes:
        root_causes_df = pd.DataFrame(root_causes, columns=["Timestamp", "Device", "Root Cause"])

        # Add an interactive filter for devices
        device_filter = st.multiselect("Filter by Device", options=root_causes_df["Device"].unique(), default=None)
        filtered_data = root_causes_df[root_causes_df["Device"].isin(device_filter)] if device_filter else root_causes_df
        st.dataframe(filtered_data)

        csv = filtered_data.to_csv(index=False)
        st.download_button("Download Root Cause Analysis", csv, "root_cause_analysis.csv")

        # AI Explanation for Root Cause Analysis
        explanation_limit = 5
        limited_root_causes = filtered_data.head(explanation_limit).values.tolist()
        prompt = (
            "The following root causes were identified from the logs:\n" +
            "\n".join([f"{row[0]} - {row[1]}: {row[2]}" for row in limited_root_causes]) +
            f"\n(Showing only the first {explanation_limit} root causes).\n" +
            "Explain the root causes in simple terms that a non-technical audience can understand."
        )
        explanation = get_ai_explanation(prompt, API_KEY, max_tokens=400)
        st.markdown(f"**AI Explanation:** {explanation}")
    else:
        st.warning("No root causes detected in the logs.")

elif option == "Visualizations":
    st.title("Visualizations")
    chart_type = st.selectbox("Select chart type:", ["Bar Chart", "Pie Chart", "Time Series Plot", "Heatmap"])
    log_summary_chart = telecom_logs["log_level"].value_counts().reset_index()
    log_summary_chart.columns = ["log_level", "count"]

    if chart_type == "Bar Chart":
        # Bar Chart
        bar_chart = alt.Chart(log_summary_chart).mark_bar().encode(
            x="log_level",
            y="count",
            color="log_level"
        ).properties(title="Log Statistics")
        st.altair_chart(bar_chart, use_container_width=True)

        # AI Explanation for Bar Chart
        prompt = (
            f"The bar chart shows the distribution of log levels as follows: "
            f"INFO: {log_summary_chart.loc[log_summary_chart['log_level'] == 'INFO', 'count'].values[0]}, "
            f"ERROR: {log_summary_chart.loc[log_summary_chart['log_level'] == 'ERROR', 'count'].values[0]}, "
            f"WARNING: {log_summary_chart.loc[log_summary_chart['log_level'] == 'WARNING', 'count'].values[0]}. "
            "Explain this chart in simple terms for a non-technical audience."
        )
        explanation = get_ai_explanation(prompt, API_KEY, max_tokens=400)
        st.write(f"**AI Explanation:** {explanation}")

    elif chart_type == "Pie Chart":
        # Pie Chart
        pie_chart = alt.Chart(log_summary_chart).mark_arc().encode(
            theta="count",
            color="log_level",
            tooltip=["log_level", "count"]
        ).properties(title="Log Statistics")
        st.altair_chart(pie_chart, use_container_width=True)

        # AI Explanation for Pie Chart
        prompt = (
            f"The pie chart shows the proportion of log levels: "
            f"INFO: {log_summary_chart.loc[log_summary_chart['log_level'] == 'INFO', 'count'].values[0]}, "
            f"ERROR: {log_summary_chart.loc[log_summary_chart['log_level'] == 'ERROR', 'count'].values[0]}, "
            f"WARNING: {log_summary_chart.loc[log_summary_chart['log_level'] == 'WARNING', 'count'].values[0]}. "
            "Explain this chart in simple terms for a non-technical audience."
        )
        explanation = get_ai_explanation(prompt, API_KEY, max_tokens=400)
        st.write(f"**AI Explanation:** {explanation}")

    elif chart_type == "Time Series Plot":
        # Time Series Plot
        telecom_logs["timestamp"] = pd.to_datetime(telecom_logs["timestamp"])
        time_series_data = telecom_logs.groupby(["timestamp", "log_level"]).size().reset_index(name="count")
        time_series_plot = px.line(
            time_series_data, 
            x="timestamp", 
            y="count", 
            color="log_level", 
            title="Time Series of Log Levels",
            labels={"count": "Log Count", "timestamp": "Time"}
        )
        st.plotly_chart(time_series_plot, use_container_width=True)

        # AI Explanation for Time Series Plot
        prompt = (
            f"The time series plot shows trends in log levels (INFO, WARNING, ERROR) over time. "
            "Provide insights about any significant patterns or spikes in the data."
        )
        explanation = get_ai_explanation(prompt, API_KEY, max_tokens=400)
        st.write(f"**AI Explanation:** {explanation}")

    elif chart_type == "Heatmap":
        # Heatmap
        heatmap_data = telecom_logs.pivot_table(
            index="device",
            columns="log_level",
            values="timestamp",
            aggfunc="count",
            fill_value=0
        )
        heatmap = px.imshow(
            heatmap_data,
            title="Heatmap of Log Levels Across Devices",
            labels={"color": "Log Count"}
        )
        st.plotly_chart(heatmap, use_container_width=True)

        # AI Explanation for Heatmap
        prompt = (
            "The heatmap shows the distribution of log levels (INFO, WARNING, ERROR) across devices. "
            "Explain any noticeable patterns or anomalies in the data."
        )
        explanation = get_ai_explanation(prompt, API_KEY, max_tokens=400)
        st.write(f"**AI Explanation:** {explanation}")


elif option == "Chatbot":
    st.title("Chat with AI")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("Ask a question about the logs:", placeholder="Ask about anomaly detection, root cause analysis, etc.")
    
    if user_input:
        # Detect language and translate if necessary
        translated_input, detected_language = translate_query(user_input)

        # Display user input in chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get AI response
        ai_response = get_enhanced_chatbot_response(translated_input, API_KEY)
        if detected_language != "en":
            ai_response = translator.translate(ai_response, src="en", dest=detected_language).text

        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
    # Display conversation
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write(f"**User:** {message['content']}")
        else:
            st.write(f"**AI:** {message['content']}")

    st.markdown("### Suggested Questions")
    st.markdown("- What is anomaly detection?")
    st.markdown("- How does root cause analysis work?")
    st.markdown("- Can you summarize the logs?")

    st.markdown("### Feedback")
    feedback = st.radio("Was this response helpful?", ("Yes", "No"), horizontal=True)
    if feedback:
        st.write("Thank you for your feedback!")

elif option == "Conclusion":
    st.title("Conclusion")
    st.markdown("""
        **Mystique** simplifies telecom log analysis with AI-powered tools.
        - 🚀 Reliable anomaly detection
        - 📈 Actionable insights
        - 🔍 Accurate root cause analysis
    """)
    st.write("Team Members: Kevin T, Shifana Azeem, Delicia Rachel Origanti, Pattan Eshaan Ahmed Khan")