# Decision Log
## Monday.com Business Intelligence Agent — Skylark Drones

---

### 1. LLM Choice — Groq with LLaMA 3.3-70B

Groq was chosen for its inference speed. On the free tier it processes around 300 tokens per second, which keeps query response time under 3 seconds even with a few hundred rows of context. LLaMA 3.3-70B handles financial reasoning and structured output well without needing fine-tuning.

OpenAI GPT-4o was considered but rejected due to cost and the latency overhead of streaming on a shared API.

---

### 2. Frontend — Streamlit

Streamlit was chosen because it allows a production-ready conversational interface with Python alone. Session state handles multi-turn chat natively. The alternative was FastAPI with a React frontend, which would have consumed most of the available time on boilerplate rather than the actual agent logic.

---

### 3. Monday.com Integration — Direct GraphQL API

Every query fires a live POST to `https://api.monday.com/v2`. No caching, no preloading. The agent uses paginated `items_page` queries with cursor-based pagination to handle boards with hundreds of items reliably.

MCP was noted as a bonus in the brief. The direct GraphQL approach was preferred because it is simpler to debug, gives full control over error handling, and does not require additional infrastructure.

---

### 4. Data Context Strategy — Computed Summary, Not Raw CSV

Initial implementation sent the full CSV to the LLM context. This caused a 413 token limit error on Groq's free tier (17,000 tokens requested against a 12,000 limit). The fix was to compute aggregated summaries — counts, totals, breakdowns by sector and status — and send only those to the LLM. This keeps the context under 4,000 tokens per query while still giving the model everything it needs to answer BI questions accurately.

---

### 5. Two-Step Agent Pipeline

Step one: a lightweight Groq call classifies the query intent — work orders, deals, or both. This avoids always fetching both boards when only one is needed.

Step two: the relevant board data is fetched live, cleaned, summarized, and sent to the LLM along with the original question and recent conversation history.

---

### 6. Data Cleaning Decisions

The data had several known issues that required explicit handling:

- Null variants: empty string, "N/A", "null", "NULL", "None", "--" all mapped to `pd.NA`
- Columns that were 100% null dropped automatically
- Header-row duplicates in the Deals board removed by checking if a cell value equals its column name
- Billing status typo ("BIlled") normalized to "Billed"
- Currency strings with Rs., commas, and mixed float/int types all parsed through a single `parse_currency()` function
- Pandas `NA` boolean evaluation error fixed by routing all value checks through a `safe_str()` wrapper that calls `pd.isna()` before any string or boolean operation

---

### 7. Deployment — Streamlit Community Cloud

The app is deployed on Streamlit Community Cloud. It connects to a public GitHub repository. Every push to main triggers an automatic redeploy. This gives a permanent live URL with no infrastructure cost.

API keys are embedded directly in the code per the assignment requirement. In production these would be moved to environment variables.

---

### Note on API Usage and Deployment Constraints

This prototype was developed under limited time and cost constraints. Due to the need for a fully functional live demonstration and the absence of a paid hosting or secret-management environment, the required API credentials are temporarily included in the application configuration.

All APIs used in this project operate under **free-tier / trial access**, including:

* Groq LLM API (free trial)
* monday.com developer workspace (free plan)

These credentials are **non-production, limited-scope keys** created only for evaluation purposes. They do not provide access to any sensitive or commercial data.

In a production environment, all secrets would be securely managed using environment variables or a secret management service instead of being stored in the source code.

