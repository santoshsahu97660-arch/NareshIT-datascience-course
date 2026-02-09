import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama
from PIL import Image

# -----------------------------
# AI INSIGHTS USING OLLAMA
# -----------------------------
def generate_ai_insights(df_summary):
    prompt = f"""
You are a Data Analyst.

Analyze the following dataset summary and provide:
1. Key observations
2. Patterns and trends
3. Data quality issues
4. Business or analytical insights (if applicable)

Dataset Summary:
{df_summary}
"""
    response = ollama.chat(
        model="gemma3:270m",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# -----------------------------
# GENERATE VISUALIZATIONS
# -----------------------------
def generate_visualizations(df):
    images = []

    # Histograms for numeric columns
    for col in df.select_dtypes(include=["number"]).columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()

        path = f"{col}_distribution.png"
        plt.savefig(path)
        plt.close()

        images.append(Image.open(path))

    # Correlation Heatmap
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        plt.figure(figsize=(8, 5))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )
        plt.title("Correlation Heatmap")
        plt.tight_layout()

        path = "correlation_heatmap.png"
        plt.savefig(path)
        plt.close()

        images.append(Image.open(path))

    return images


# -----------------------------
# MAIN EDA FUNCTION
# -----------------------------
def eda_analysis(file_path):
    try:
        df = pd.read_csv(file_path)

        # Handle missing values
        for col in df.select_dtypes(include=["number"]).columns:
            df[col].fillna(df[col].median(), inplace=True)

        for col in df.select_dtypes(include=["object"]).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

        summary = df.describe(include="all").to_string()
        missing_values = df.isnull().sum().to_string()

        insights = generate_ai_insights(summary)
        images = generate_visualizations(df)

        report = f"""
✅ DATA LOADED SUCCESSFULLY

📊 DATA SUMMARY:
{summary}

🧹 MISSING VALUES:
{missing_values}

🤖 AI-GENERATED INSIGHTS:
{insights}
"""

        return report, images

    except Exception as e:
        return f"❌ Error occurred:\n{str(e)}", []


# -----------------------------
# GRADIO UI
# -----------------------------
demo = gr.Interface(
    fn=eda_analysis,
    inputs=gr.File(type="filepath", label="Upload CSV File"),
    outputs=[
        gr.Textbox(label="EDA Report", lines=25),
        gr.Gallery(label="Data Visualizations")
    ],
    title="📊 LLM-Powered Exploratory Data Analysis (EDA)",
    description=(
        "Upload any CSV file to automatically generate EDA reports, "
        "AI insights, and visualizations using Ollama + Gradio."
    )
)

# -----------------------------
# LAUNCH APP
# -----------------------------
demo.launch(share=True)
