import os
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql.types import StructType, StructField, StringType
from jobspy import scrape_jobs
import PyPDF2
import pandas as pd
import re
from groq import Groq
import json

spark = SparkSession.builder.appName("ResumeBasedJobRecommendation").getOrCreate()
client = Groq(api_key="YOUR_API_KEY")

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_resume_data_with_groq(resume_text):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""Extract the following information from the resume text:
- List of professional skills
- Current location
- Country

and make sure you provide a suitable job title considering all the skills

Return the response as a VALID JSON with these exact keys:
{{
    "skills": ["skill1", "skill2"],
    "location": "city name",
    "country": "country name"
    "suitable_job" : "job_title"
}}

Resume Text:
{resume_text}"""
                }
            ],
            model="llama3-8b-8192",
        )

        extracted_data_str = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', extracted_data_str, re.DOTALL)

        if json_match:
            extracted_data_str = json_match.group(0)

        extracted_data = json.loads(extracted_data_str)

        return extracted_data
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON from GroqAI: {e}")
        return {
            "skills": ["software engineering"],
            "location": "bangalore",
            "country": "india",
            "suitable_job" : "software Developer engineer"
        }
    except Exception as e:
        st.error(f"Unexpected error using GroqAI: {e}")
        return {
            "skills": ["software engineering"],
            "location": "bangalore",
            "country": "india",
            "suitable_job" : "SDE"

        }

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    st.title("Resume-Based Job Recommendation System")

    uploaded_file = st.file_uploader("Upload your resume (PDF format only):", type=["pdf"])

    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)

        if not resume_text:
            st.error("Could not extract text from the PDF. Please check the file.")
            return

        st.info("Extracting skills, location, and country using GroqAI...")
        resume_data = extract_resume_data_with_groq(resume_text)

        skills = resume_data.get("skills", ["software engineer"])
        location = resume_data.get("location", "bangalore")
        country_indeed = resume_data.get("country", "india")
        suitable_job = resume_data.get("suitable_job", "Software Developer Engineer")

        st.write(f"**Identified Skills:** {', '.join(skills)}")
        st.write(f"**Identified Location:** {location}")
        st.write(f"**Identified Country:** {country_indeed}")
        st.write(f"**Best Suitable Job identified:** {suitable_job}")

        if(country_indeed == "IN"):
          country_indeed = "India"

        st.info("Fetching job postings...")
        jobs = scrape_jobs(
            search_term=suitable_job,
            location=location,
            results_wanted=50,
            hours_old=72,
            country_indeed=country_indeed
        )
        st.success(f"Fetched {len(jobs)} jobs!")

        jobs_df = pd.DataFrame(jobs)

        columns_to_select = [
            'title' if 'title' in jobs_df.columns else 'job_title',
            'company' if 'company' in jobs_df.columns else 'company_name',
            'location',
            'description',
            'job_url' if 'job_url' in jobs_df.columns else 'url'
        ]

        columns_to_select = [col for col in columns_to_select if col in jobs_df.columns]

        jobs_df = jobs_df.loc[:, columns_to_select]

        schema = StructType([
            StructField("job_title", StringType(), True),
            StructField("company_name", StringType(), True),
            StructField("location", StringType(), True),
            StructField("description", StringType(), True),
            StructField("url", StringType(), True),
        ])

        spark_jobs_df = spark.createDataFrame(jobs_df, schema=schema)

        resume_data = [{"resume_id": 1, "content": " ".join(skills)}]
        resume_df = spark.createDataFrame(resume_data)

        tokenizer = Tokenizer(inputCol="description", outputCol="words")
        hashed_tf = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
        idf = IDF(inputCol="rawFeatures", outputCol="features")

        tokenized_jobs = tokenizer.transform(spark_jobs_df)
        tf_jobs = hashed_tf.transform(tokenized_jobs)
        idf_model_jobs = idf.fit(tf_jobs)
        tfidf_jobs = idf_model_jobs.transform(tf_jobs)

        tokenizer_resume = Tokenizer(inputCol="content", outputCol="words")
        tokenized_resume = tokenizer_resume.transform(resume_df)
        tf_resume = hashed_tf.transform(tokenized_resume)
        idf_model_resume = idf.fit(tf_resume)
        tfidf_resume = idf_model_resume.transform(tf_resume)

        resume_vector = tfidf_resume.select("features").collect()[0][0]
        job_scores = tfidf_jobs.rdd.map(
            lambda row: (row["job_title"], row["company_name"], row["location"], row["url"])).take(5)

        st.write("## Top Job Recommendations")
        if job_scores:
            for job in job_scores:
                st.write(f"**Job Title:** {job[0]}")
                st.write(f"**Company:** {job[1]}")
                st.write(f"**Location:** {job[2]}")
                st.write(f"**Job URL:** [Click Here]({job[3]})")
                st.write("---")
        else:
            st.warning("No recommendations found. Try modifying your resume or search parameters.")

if __name__ == "__main__":
    main()
