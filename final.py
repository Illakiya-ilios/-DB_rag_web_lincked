import os
import boto3
from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain.tools import tool
from fastapi import FastAPI
from pydantic import BaseModel

# ========================
# Load environment
# ========================
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "12.25.11.2")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "stm_db")
DB_USER = os.getenv("DB_USER", "app_user")
DB_PASS = os.getenv("DB_PASS", "fanofabds84")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

# ========================
# AWS Bedrock Client
# ========================
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION
)

# ========================
# Database Connection
# ========================
uri = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
db = SQLDatabase.from_uri(uri, include_tables=["students"])

# Create the SQL execution tool
sql_tool = QuerySQLDatabaseTool(db=db)


# ========================
# TOOL for executing queries
# ========================
@tool
def run_sql(query: str):
    """Execute SQL query on the database and return results."""
    return sql_tool.invoke(query)


# ========================
# LLM Model
# ========================
llm = ChatBedrock(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    client=bedrock
)


# ========================
# SQL Generator Prompt
# ========================
sql_prompt = PromptTemplate.from_template(
"""
You are an expert SQL assistant who converts user questions into SQL queries
to retrieve data from a relational database.

IMPORTANT:
- Return ONLY raw SQL query
- DO NOT explain anything
- DO NOT include ```sql or markdown
- DO NOT include any text before or after SQL
- Output must start directly with SELECT
- If you include anything else, the response is invalid

--------------------------------------------------

Database schema:
{schema}

The database contains multiple tables including (but not limited to):

- students → student personal + academic details
- schools → school information
- school_users → school staff (admins/principals)
- admin_profile → admin user details
- student_transfer → transfer certificate workflow
- tc → issued transfer certificates
- tc_files → uploaded documents
- student_otp / admin_otp / superadmin_otp → OTP records (DO NOT query unless explicitly asked)

Key relationships:

- school_users.school_id → schools.school_id
- students.udise_code ↔ schools.udise
- tc.student_uuid → students.uuid
- tc_files.ticket_id → tc.tc_id

--------------------------------------------------

INSTRUCTIONS:

1. Understand user intent:

- Student-related (details, TC, personal info)
- School-related (schools, count, district)
- Admin/staff-related
- Transfer/TC-related
- General queries (counts, lists)

--------------------------------------------------

2. QUERY TYPE HANDLING:

A) GENERAL QUERIES (NO identification required):
- "how many schools are there"
- "how many students"
- "list schools"
- "give a school name"

→ ALWAYS generate SQL

B) SPECIFIC LOOKUPS (IDENTIFICATION required):

For STUDENTS:
- Require at least ONE identifier:
  (name, UUID, mobile, etc.)
- If missing → return exactly:
  Please provide student details such as name, UUID, or registered mobile number.

For ADMINS:
- Require email, employee_id, or name

--------------------------------------------------

3. SQL RULES:

- Use MySQL syntax ONLY
- Use COUNT(*) for counts
- Use LIMIT 1 when only one result needed
- Use LIKE for text search (NOT ILIKE)
- Use JOINs when needed
- NEVER expose sensitive fields:
  password, password_hash, otp_code

--------------------------------------------------

4. OUTPUT RULE:

Return ONLY ONE of:
- A valid SQL query starting with SELECT
OR
- A clarification message (ONLY if required)

--------------------------------------------------
EXAMPLES:

Q: how many schools are there
SELECT COUNT(*) AS total_schools FROM schools;

Q: how many students are there
SELECT COUNT(*) AS student_count FROM students;

Q: give me a school name
SELECT school_name FROM schools LIMIT 1;

Q: list schools in Chennai
SELECT school_name, district FROM schools
WHERE district LIKE '%Chennai%';

Q: get student details for Vihaan Aarav Jain
SELECT s.uuid, s.student_name, s.primary_mobile, s.school_name
FROM students s
WHERE s.student_name LIKE '%Vihaan Aarav Jain%'
LIMIT 1;

Q: show TC status for student Vihaan Aarav Jain
SELECT t.tc_id, t.student_name, t.tc_status, t.updated_at
FROM tc t
WHERE t.student_name LIKE '%Vihaan Aarav Jain%'
LIMIT 1;

--------------------------------------------------

Question: {question}
"""
)

def clean_sql(query: str):
    # Remove markdown
    query = query.replace("```sql", "").replace("```", "")

    # Remove explanations (keep only SELECT part)
    if "SELECT" in query.upper():
        query = query[query.upper().find("SELECT"):]

    return query.strip()

generate_query = sql_prompt | llm | StrOutputParser()


# ========================
# Answer Generation Prompt
# ========================
answer_prompt = PromptTemplate.from_template(
"""
You are a helpful assistant.

Your task is to convert SQL results into a clear, natural language response.
Do NOT mention SQL, queries, tables, or database terms.

--------------------------------------------------

RULES:

1. If the SQL result is empty:
→ "No record found for the given details."

2. If the previous step asked for more details:
→ Repeat the same message politely.

3. If data is found:

- For student queries:
  Provide clear details like name, school, status, etc.

- For school queries:
  Mention school name, district, type, and student volume.

- For transfer/TC queries:
  Mention TC ID, status, and relevant updates.

- For admin/staff queries:
  Mention name, role, school, and contact if available.

4. Keep response:
- Natural
- Concise
- Human-friendly

--------------------------------------------------

EXAMPLES:

Input result:
(student_name = "Vihaan Aarav Jain", tc_status = "CURRENT_PRINCIPAL_APPROVED")

Output:
"Vihaan Aarav Jain's transfer certificate is currently approved by the principal."

---

Input result:
(school_name = "Silvassa", district = "Chennai", student_volume = 3900)

Output:
"The school Silvassa is located in Chennai and has approximately 3900 students."

--------------------------------------------------

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer:
"""
)

format_answer = answer_prompt | llm | StrOutputParser()


# ========================
# MAIN CHAIN
# ========================
chain = (
    RunnablePassthrough()
    .assign(
        schema=lambda x: db.get_table_info()
    )
    .assign(
        raw_query=lambda x: generate_query.invoke(
            {"question": x["question"], "schema": x["schema"]}
        )
    )
    .assign(
        query=lambda x: clean_sql(x["raw_query"])
    )
    .assign(
        result=lambda x: (
            "Please provide more identifying details such as your student ID, registered mobile number, or full name."
            if "Please provide" in x["query"]
            else run_sql.invoke(x["query"])
        )
    )
    | format_answer
)


# ========================
# FASTAPI ENDPOINT
# =======================

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        answer = chain.invoke({"question": req.question})
        return {
            "question": req.question,
            "answer": answer
        }
    except Exception as e:
        return {
            "error": str(e)
        }


# ========================
# RUN SERVER
# ========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8333)