# Monday.com BI Agent — Skylark Drones

A conversational business intelligence agent that answers questions about work orders and sales pipeline by fetching live data from monday.com boards.

---

## What it does

You ask a question in plain English. The agent figures out which monday.com board to query, fetches the data live, cleans it, and returns a structured answer with key numbers, business insight, and data quality notes. Every query shows the full execution trace — which API was called, how many items were returned, and how long it took.

---

## Tech Stack

- Python, Streamlit
- Groq API — LLaMA 3.3-70B
- Monday.com GraphQL API v2
- Pandas

---

## Local Setup

### 1. Clone or extract the project

Place the project folder somewhere on your machine and open a terminal inside it.

### 2. Create a virtual environment

Windows:
```
python -m venv venv
```

Mac / Linux:
```
python3 -m venv venv
```

This creates a folder named `venv` inside the project directory.

### 3. Activate the virtual environment

Windows (Command Prompt):
```
venv\Scripts\activate
```

Windows (PowerShell):
```
venv\Scripts\Activate.ps1
```

If PowerShell shows a permission error, run this once and then try again:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Mac / Linux:
```
source venv/bin/activate
```

Once activated, your terminal prompt will show `(venv)` at the start.

### 4. Install dependencies

```
pip install -r requirements.txt
```

### 5. Run the app

```
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## Deploy on Streamlit Cloud

1. Push this folder to a public GitHub repository
2. Go to share.streamlit.io and sign in with GitHub
3. Select your repository, set the main file as `app.py`, and click Deploy
4. Your live URL will be ready in about 2 minutes

---

## Monday.com Boards

Two boards are required:

- Work Orders board — imported from `Work_Order_Tracker_Data.xlsx`
- Deals board — imported from `Deal_funnel_Data.xlsx`

To import: open monday.com, create a new board, click the three-dot menu, select Import, and upload the Excel file.

The board IDs are in `app.py` at lines 13 and 14. You can get a board ID from the URL when the board is open — the number after `/boards/` in the address bar.

---

## Files

```
app.py                        main application
requirements.txt              dependencies
DECISION_LOG.md               technical decisions
Work_Order_Tracker_Data.xlsx  source data
Deal_funnel_Data.xlsx         source data
.streamlit/config.toml        theme config
```

---

## Sample Questions

- How is our pipeline looking for the energy sector?
- What is our billing and collection efficiency?
- Which high-probability deals are still open?
- Show accounts receivable breakdown by sector
- Which work orders are stuck or on hold?
- Compare pipeline value against executed work order value

## Input/Output Sample
<img width="1810" height="807" alt="Screenshot 2026-02-27 131009" src="https://github.com/user-attachments/assets/09b6e8ee-17cb-4351-b8cf-8c95bc0837c4" />

<img width="1870" height="672" alt="Screenshot 2026-02-27 124221" src="https://github.com/user-attachments/assets/4a1fdd1f-68f3-497e-872b-1b0380deb7e0" />


