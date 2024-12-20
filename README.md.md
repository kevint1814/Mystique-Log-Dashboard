Mystique - Telecom Network Log Analysis Dashboard
Mystique is a cutting-edge tool designed to assist telecom operators in analyzing network logs, detecting anomalies, summarizing key events, and identifying root causes of network outages using AI and machine learning. The interactive dashboard ensures fast, reliable, and user-friendly network management.

Features
Dataset Visualization: Preview telecom log datasets, including timestamps, devices, log levels, and messages.
Anomaly Detection: Identify anomalies in network logs using machine learning (Isolation Forest).
AI-Powered Summarization: Summarize log entries with clear, actionable insights powered by GroqCloud AI.
Root Cause Analysis: Pinpoint potential causes of network issues with AI-powered analysis.
Dynamic Visualizations: Generate bar and pie charts to analyze log statistics interactively.
Professional Conclusion: Get a summarized analysis of the entire log processing workflow.
System Requirements
Operating System: Windows, macOS, or Linux
Python Version: Python 3.8 or above
Libraries: Listed in requirements.txt
Installation Guide
1. Clone the Repository
bash
Copy code
git clone https://github.com/your-repo/mystique.git
cd mystique
2. Install Dependencies
Ensure all required libraries are installed:

bash
Copy code
pip install -r requirements.txt
3. Run the Application
Start the Streamlit dashboard:

bash
Copy code
streamlit run mystique_dashboard.py
Executable Version
If you prefer to use the standalone executable:

Download the mystique_dashboard.exe file from the dist/ folder.
Double-click to launch the application.
The dashboard will open in your default web browser.
Usage Instructions
1. View the Dataset
Preview the telecom logs to get an overview of the data.
2. Detect Anomalies
Adjust the anomaly detection sensitivity using the slider.
View detected anomalies in a table format for further inspection.
3. Summarize Logs
Use AI to summarize a specific number of log entries.
Adjust the number of logs using the slider.
4. Root Cause Analysis
Filter anomalies by device to pinpoint issues.
View results in an easy-to-read table.
5. Visualizations
Choose between bar and pie charts to visualize log statistics.
Read AI-generated explanations for chart insights.
6. Conclusion
Review a detailed analysis summarizing the tool’s findings and insights.
Technical Details
Core Technologies
Python: Backend logic and data processing.
Streamlit: Interactive dashboard interface.
Machine Learning: Isolation Forest for anomaly detection.
GroqCloud API: AI-powered log summarization and insights.
Key Files
mystique_dashboard.py: Main application script.
Clear_Anomalies_Telecom_Logs_Dataset.csv: Sample dataset for testing.
requirements.txt: Python dependencies.
Contributors
Kevin T
Shifana Azeem
Delicia Rachel Origanti
Pataan Eshaan Ahmed Khan
Future Enhancements
Add real-time log ingestion capabilities.
Expand the AI summarization to handle larger datasets.
Integrate multi-user authentication and role-based access.
License
This project is licensed under the MIT License.