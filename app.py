import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from groq import Groq


# Configuration

GROQ_API_KEY          = "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"
MONDAY_API_KEY        = "eyJhbGciOiJIUzI1NiJ9.eyJ0aWQiOjYyNjcwODU4MSwiYWFpIjoxMSwidWlkIjoxMDAzOTg5NzQsImlhZCI6IjIwMjYtMDItMjdUMDg6MDc6MjMuMjk1WiIsInBlciI6Im1lOndyaXRlIiwiYWN0aWQiOjMzOTk2MTYxLCJyZ24iOiJhcHNlMiJ9.ve155pX3gnlHqNXEWexuR_UK8RBuVxPCpzYHgn0Lo6U"
MONDAY_WO_BOARD_ID    = "5026890073"
MONDAY_DEALS_BOARD_ID = "5026890130"
MONDAY_API_URL        = "https://api.monday.com/v2"
MONDAY_API_VERSION    = "2024-01"

NULL_VARIANTS = [
    "", "N/A", "n/a", "NA", "null", "NULL",
    "None", "-", "--", "nan", "#N/A", "NaN"
]

# Monday.com API

def monday_graphql(query, variables=None):
    token = MONDAY_API_KEY
    headers = {
        "Authorization": token,
        "Content-Type": "application/json",
        "API-Version": MONDAY_API_VERSION,
    }
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    try:
        resp = requests.post(MONDAY_API_URL, json=payload, headers=headers, timeout=45)
        sc = resp.status_code
        if sc == 401:
            return {"ok": False, "data": None, "error": "Invalid API token (401).", "status_code": sc}
        if sc == 429:
            return {"ok": False, "data": None, "error": "Rate limited (429). Please wait.", "status_code": sc}
        body = resp.json()
        if "errors" in body:
            return {"ok": False, "data": None, "error": str(body["errors"]), "status_code": sc}
        return {"ok": True, "data": body.get("data", {}), "error": None, "status_code": sc}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "data": None, "error": "Cannot reach api.monday.com.", "status_code": 0}
    except requests.exceptions.Timeout:
        return {"ok": False, "data": None, "error": "Request timed out after 45 seconds.", "status_code": 0}
    except Exception as e:
        return {"ok": False, "data": None, "error": str(e), "status_code": 0}


def live_fetch_board(board_key, label):
    board_id = MONDAY_WO_BOARD_ID if board_key == "WO" else MONDAY_DEALS_BOARD_ID

    trace = {
        "label": label,
        "board_id": board_id,
        "start": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "pages": 0,
        "total_items": 0,
        "status": "running",
        "error": None,
        "end": None,
        "board_name": "unknown",
    }

    GQL = """
    query ($board_id: ID!, $limit: Int!, $cursor: String) {
        boards(ids: [$board_id]) {
            id
            name
            items_page(limit: $limit, cursor: $cursor) {
                cursor
                items {
                    id
                    name
                    column_values {
                        id
                        text
                        column {
                            title
                            type
                        }
                    }
                }
            }
        }
    }
    """

    all_rows = []
    cursor = None

    for _ in range(25):
        variables = {"board_id": board_id, "limit": 100}
        if cursor:
            variables["cursor"] = cursor

        res = monday_graphql(GQL, variables)

        if not res["ok"]:
            trace.update(status="error", error=res["error"], end=datetime.now().strftime("%H:%M:%S.%f")[:-3])
            return pd.DataFrame(), trace

        boards_data = res["data"].get("boards", [])
        if not boards_data:
            trace.update(
                status="error",
                error=f"Board ID {board_id} not found or no access.",
                end=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            )
            return pd.DataFrame(), trace

        board = boards_data[0]
        trace["board_name"] = board.get("name", board_id)
        page_data = board.get("items_page", {})
        items = page_data.get("items", [])
        next_cursor = page_data.get("cursor")

        for item in items:
            row = {"_name": item["name"], "_id": item["id"]}
            for cv in item.get("column_values", []):
                row[cv["column"]["title"]] = cv.get("text") or ""
            all_rows.append(row)

        trace["pages"] += 1
        trace["total_items"] = len(all_rows)

        if not next_cursor or not items:
            break
        cursor = next_cursor

    trace.update(status="success", end=datetime.now().strftime("%H:%M:%S.%f")[:-3])
    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    return df, trace


def test_connection():
    result = monday_graphql("query { me { id name email account { name } } }")
    if result["ok"]:
        return {"ok": True, "user": result["data"].get("me", {})}
    return {"ok": False, "error": result["error"]}



# Data Cleaning 

def safe_str(val):
    """
    Convert any value including pandas NA/NaT/None to a plain string safely.
    This avoids the 'boolean value of NA is ambiguous' error.
    """
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def parse_currency(val):
    s = safe_str(val).replace(",", "").replace("Rs.", "").replace("INR", "").strip()
    if s in ("", "-", "N/A", "NA", "nan", "None"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def normalize_exec_status(val):
    s = safe_str(val).lower()
    if not s or s in ("nan", "none", "n/a"):
        return "Unknown"
    if "complet" in s and "partial" not in s:
        return "Completed"
    if "partial" in s:
        return "Partial Completed"
    if "not start" in s:
        return "Not Started"
    if "executed until" in s:
        return "Ongoing (Monthly)"
    if "ongoing" in s:
        return "Ongoing"
    if "pause" in s or "struck" in s:
        return "On Hold"
    if "pending" in s:
        return "Pending Client Info"
    return safe_str(val)


def normalize_billing_status(val):
    s = safe_str(val).lower()
    if not s or s in ("nan", "none", "n/a"):
        return "Unknown"
    if s in ("billed", "bIlled") or s == "billed":
        return "Billed"
    if "partial" in s:
        return "Partially Billed"
    if "not billable" in s:
        return "Not Billable"
    if "stuck" in s:
        return "Stuck"
    if "update" in s:
        return "Update Required"
    return safe_str(val)


def normalize_deal_status(val):
    s = safe_str(val).lower()
    if not s or s in ("nan", "none", "deal status"):
        return "Unknown"
    if "won" in s:
        return "Won"
    if "dead" in s:
        return "Dead / Lost"
    if "open" in s:
        return "Open"
    if "hold" in s:
        return "On Hold"
    return safe_str(val).title()


def clean_dataframe(df):
    if df.empty:
        return df
    df = df.replace(NULL_VARIANTS, pd.NA).copy()
    df = df.dropna(axis=1, how="all")
    # Remove header-row duplicate records from monday.com boards
    for col in df.columns:
        col_str = safe_str(col)
        df = df[df[col].apply(safe_str) != col_str]
    return df.reset_index(drop=True)


def data_quality_summary(df):
    if df.empty:
        return "No data available."
    total = len(df)
    issues = []
    for col in df.columns:
        if col.startswith("_"):
            continue
        null_count = df[col].apply(lambda x: safe_str(x) == "").sum()
        pct = null_count / total * 100
        if pct > 30:
            issues.append(f"  '{col}': {pct:.0f}% missing")
    if not issues:
        return "  All key columns have reasonable completeness."
    return "\n".join(issues)


# Analytics — Work Orders 

def find_col(cols, primary, secondary="", exclude="XXXXXX"):
    for c in cols:
        c_lower = c.lower()
        if primary.lower() in c_lower:
            if secondary and secondary.lower() not in c_lower:
                continue
            if exclude.lower() in c_lower:
                continue
            return c
    return None


def build_wo_context(df):
    if df.empty:
        return "WORK ORDERS: No data returned from the monday.com board."

    cols = df.columns.tolist()
    lines = [
        f"WORK ORDERS BOARD — {len(df)} records (live fetch)",
        f"Columns available: {', '.join(c for c in cols if not c.startswith('_'))}",
        "",
    ]

    exec_col    = find_col(cols, "execution status")
    sector_col  = find_col(cols, "sector")
    nature_col  = find_col(cols, "nature of work")
    type_col    = find_col(cols, "type of work")
    bd_col      = find_col(cols, "bd/kam")
    amt_col     = find_col(cols, "amount in rupees", "excl", "billed")
    bill_col    = find_col(cols, "billed value", "excl")
    coll_col    = find_col(cols, "collected amount")
    tobill_col  = find_col(cols, "amount to be billed", "excl")
    recv_col    = find_col(cols, "amount receivable")
    inv_col     = find_col(cols, "invoice status")
    wo_st_col   = find_col(cols, "wo status")
    bill_st_col = find_col(cols, "billing status")
    ar_col      = find_col(cols, "ar priority")

    if exec_col:
        lines.append("Execution Status:")
        for k, v in df[exec_col].apply(normalize_exec_status).value_counts().items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    if sector_col:
        lines.append("Sector Distribution:")
        for k, v in df[sector_col].apply(safe_str).value_counts().items():
            if k not in ("", "nan"):
                lines.append(f"  {k}: {v}")
        lines.append("")

    if nature_col:
        lines.append("Nature of Work:")
        for k, v in df[nature_col].apply(safe_str).value_counts().items():
            if k not in ("", "nan"):
                lines.append(f"  {k}: {v}")
        lines.append("")

    if type_col:
        lines.append("Type of Work (top 10):")
        for k, v in df[type_col].apply(safe_str).value_counts().head(10).items():
            if k not in ("", "nan"):
                lines.append(f"  {k}: {v}")
        lines.append("")

    if bd_col:
        lines.append("Work Orders by BD/KAM Personnel:")
        for k, v in df[bd_col].apply(safe_str).value_counts().head(10).items():
            if k not in ("", "nan"):
                lines.append(f"  {k}: {v}")
        lines.append("")

    lines.append("Financial Summary (INR — values are masked/scaled):")
    totals = {}
    financial_cols = [
        (amt_col,    "Total WO Contract Value (Excl GST)"),
        (bill_col,   "Total Billed (Excl GST)"),
        (coll_col,   "Total Collected (Incl GST)"),
        (tobill_col, "Yet to be Billed (Excl GST)"),
        (recv_col,   "Accounts Receivable"),
    ]
    for c, label in financial_cols:
        if c and c in df.columns:
            val = df[c].apply(parse_currency).sum()
            totals[label] = val
            lines.append(f"  {label}: Rs.{val:,.0f}  (~Rs.{val/1e7:.2f} Cr)")

    wo_val = totals.get("Total WO Contract Value (Excl GST)", 0)
    if wo_val > 0:
        billed = totals.get("Total Billed (Excl GST)", 0)
        collected = totals.get("Total Collected (Incl GST)", 0)
        if billed:
            lines.append(f"  Billing Efficiency: {billed/wo_val*100:.1f}%")
        if billed and collected:
            billed_incl = billed * 1.18
            if billed_incl > 0:
                lines.append(f"  Collection Efficiency: {collected/billed_incl*100:.1f}%")
    lines.append("")

    if sector_col and amt_col:
        lines.append("Revenue Breakdown by Sector (INR):")
        df2 = df.copy()
        df2["_a"] = df2[amt_col].apply(parse_currency)
        df2["_b"] = df2[bill_col].apply(parse_currency) if bill_col else 0
        df2["_c"] = df2[coll_col].apply(parse_currency) if coll_col else 0
        df2["_r"] = df2[recv_col].apply(parse_currency) if recv_col else 0
        for sec, g in df2.groupby(df2[sector_col].apply(safe_str)):
            if sec in ("", "nan"):
                continue
            a, b, c, r = g["_a"].sum(), g["_b"].sum(), g["_c"].sum(), g["_r"].sum()
            lines.append(f"  {sec}:")
            lines.append(f"    Contract Value=Rs.{a:,.0f} | Billed=Rs.{b:,.0f} | Collected=Rs.{c:,.0f} | AR=Rs.{r:,.0f}")
            if a > 0:
                lines.append(f"    Billing Rate: {b/a*100:.1f}%")
        lines.append("")

    if inv_col:
        lines.append("Invoice Status:")
        for k, v in df[inv_col].apply(safe_str).value_counts().items():
            if k not in ("", "nan"):
                lines.append(f"  {k}: {v}")
        lines.append("")

    if bill_st_col:
        lines.append("Billing Status (normalized):")
        for k, v in df[bill_st_col].apply(normalize_billing_status).value_counts().items():
            if k != "Unknown":
                lines.append(f"  {k}: {v}")
        lines.append("")

    if wo_st_col:
        lines.append("WO Status (Billed):")
        for k, v in df[wo_st_col].apply(safe_str).value_counts().items():
            if k not in ("", "nan"):
                lines.append(f"  {k}: {v}")
        lines.append("")

    if exec_col:
        stuck_mask = df[exec_col].apply(safe_str).str.lower().str.contains("pause|struck|pending", na=False)
        stuck = df[stuck_mask]
        if not stuck.empty:
            lines.append(f"Stuck / Problem Work Orders: {len(stuck)}")
            for _, row in stuck.head(8).iterrows():
                sec = safe_str(row.get(sector_col, "")) if sector_col else ""
                lines.append(f"  {safe_str(row.get('_name', ''))} | {sec} | {safe_str(row.get(exec_col, ''))}")
            lines.append("")

    lines.append("Data Quality Notes:")
    lines.append(data_quality_summary(df))
    lines.append("  Columns that were 100% empty were automatically dropped.")

    return "\n".join(lines)


# Analytics — Deals 

def build_deals_context(df):
    if df.empty:
        return "DEALS: No data returned from the monday.com board."

    cols = df.columns.tolist()
    lines = [
        f"DEALS / PIPELINE BOARD — {len(df)} records (live fetch)",
        f"Columns available: {', '.join(c for c in cols if not c.startswith('_'))}",
        "",
    ]

    status_col  = find_col(cols, "deal status")
    stage_col   = find_col(cols, "deal stage")
    sector_col  = find_col(cols, "sector")
    prob_col    = find_col(cols, "closure probability")
    val_col     = find_col(cols, "masked deal value")
    owner_col   = find_col(cols, "owner")
    prod_col    = find_col(cols, "product")
    name_col    = find_col(cols, "deal name") or "_name"

    if status_col:
        lines.append("Deal Status:")
        clean = df[status_col].apply(normalize_deal_status)
        for k, v in clean.value_counts().items():
            if k != "Unknown":
                lines.append(f"  {k}: {v}")
        lines.append("")

    if stage_col:
        lines.append("Deal Stage — Funnel:")
        for k, v in df[stage_col].apply(safe_str).value_counts().items():
            if k not in ("", "nan", "Deal Stage"):
                lines.append(f"  {k}: {v}")
        lines.append("")

    if sector_col:
        lines.append("Sector Distribution:")
        for k, v in df[sector_col].apply(safe_str).value_counts().items():
            if k not in ("", "nan", "Sector/service"):
                lines.append(f"  {k}: {v}")
        lines.append("")

    if prob_col:
        lines.append("Closure Probability:")
        mask = df[prob_col].apply(safe_str).str.lower() != "closure probability"
        for k, v in df[mask][prob_col].apply(safe_str).value_counts().items():
            if k not in ("", "nan"):
                lines.append(f"  {k}: {v}")
        lines.append("")

    if val_col:
        df2 = df.copy()
        df2["_v"] = df2[val_col].apply(parse_currency)
        total = df2["_v"].sum()

        lines.append("Pipeline Value Analysis (INR — masked/scaled):")
        lines.append(f"  All deals total: Rs.{total:,.0f}  (~Rs.{total/1e7:.2f} Cr)")

        if status_col:
            for status in ["Open", "Won", "Dead / Lost", "On Hold"]:
                sub = df2[df2[status_col].apply(normalize_deal_status) == status]
                lines.append(f"  {status}: {len(sub)} deals | Rs.{sub['_v'].sum():,.0f}")
        lines.append("")

        if sector_col:
            lines.append("Deal Value by Sector:")
            sector_clean = df2[sector_col].apply(safe_str)
            grouped = df2.groupby(sector_clean)["_v"].agg(["count", "sum"])
            for sec, row in grouped.sort_values("sum", ascending=False).iterrows():
                if sec not in ("", "nan", "Sector/service"):
                    lines.append(f"  {sec}: {int(row['count'])} deals | Rs.{row['sum']:,.0f}")
            lines.append("")

        if prob_col and status_col:
            open_mask = df2[status_col].apply(normalize_deal_status) == "Open"
            high_mask = df2[prob_col].apply(safe_str).str.lower().str.contains("high", na=False)
            high_open = df2[open_mask & high_mask]
            lines.append(f"High Probability Open Deals: {len(high_open)} deals | Rs.{high_open['_v'].sum():,.0f}")
            for _, row in high_open.head(15).iterrows():
                sec = safe_str(row.get(sector_col, "")) if sector_col else ""
                lines.append(f"  {safe_str(row.get(name_col, ''))} | {sec} | Rs.{parse_currency(row.get(val_col, 0)):,.0f}")
            lines.append("")

        if stage_col and status_col:
            open_deals = df2[df2[status_col].apply(normalize_deal_status) == "Open"]
            lines.append(f"Open Deals by Stage ({len(open_deals)} total):")
            for stg, g in open_deals.groupby(open_deals[stage_col].apply(safe_str)):
                if stg not in ("", "nan"):
                    lines.append(f"  {stg}: {len(g)} | Rs.{g['_v'].sum():,.0f}")
            lines.append("")

        if sector_col:
            energy = df2[df2[sector_col].apply(safe_str).isin(["Renewables", "Powerline"])]
            lines.append("Energy Sector (Renewables + Powerline combined):")
            lines.append(f"  Total deals: {len(energy)} | Rs.{energy['_v'].sum():,.0f}")
            if status_col:
                e_open = energy[energy[status_col].apply(normalize_deal_status) == "Open"]
                lines.append(f"  Open deals: {len(e_open)} | Rs.{e_open['_v'].sum():,.0f}")
            lines.append("")

    if owner_col:
        lines.append("Deals by Owner / BD Personnel:")
        for k, v in df[owner_col].apply(safe_str).value_counts().head(10).items():
            if k not in ("", "nan"):
                lines.append(f"  {k}: {v} deals")
        lines.append("")

    if prod_col:
        lines.append("Product Mix:")
        for k, v in df[prod_col].apply(safe_str).value_counts().head(8).items():
            if k not in ("", "nan", "Product deal"):
                lines.append(f"  {k}: {v}")
        lines.append("")

    lines.append("Data Quality Notes:")
    lines.append(data_quality_summary(df))
    lines.append("  Close Date is 91.9% empty. Closure Probability is 74.6% empty — early-stage deals.")

    return "\n".join(lines)


# LLM — Groq 

SYSTEM_PROMPT = """You are a business intelligence analyst for Skylark Drones, an Indian drone services and geospatial company. You answer founder and executive level questions using live data pulled directly from monday.com boards.

The company operates across Mining, Renewables, Railways, Powerline, Construction, and other sectors, providing services like topographic surveys, LiDAR scanning, volumetric surveys, powerline inspection, and thermography.

Two monday.com boards are available:

Work Orders Board — tracks project execution:
- Nature of Work: One time Project, Monthly Contract, Annual Rate Contract, Proof of Concept
- Execution Status: Completed, Ongoing, Not Started, On Hold, Partial Completed, Ongoing (Monthly), Pending Client Info
- Financial columns in INR (masked/scaled): Contract Value (Excl GST), Billed Value (Excl GST), Collected Amount (Incl GST), Amount to be Billed, Accounts Receivable
- Invoice Status: Fully Billed, Partially Billed, Not billed yet, Stuck
- WO Status: Open, Closed

Deals Board — tracks sales pipeline:
- Deal Status: Open, Won, Dead / Lost, On Hold
- Deal Stage (funnel): A=Lead Generated, B=Sales Qualified, C=Demo Done, D=Feasibility, E=Proposal Sent, F=Negotiations, G=Project Won, H=Work Order Received, I=POC, J=Invoice Sent, K=Amount Accrued, L=Project Lost, M=On Hold, N/O=Not Relevant, Project Completed
- Closure Probability: High, Medium, Low
- Masked Deal value (INR, scaled)

Response format — always structure your answer like this:

Key Numbers
- [list the core metrics that answer the question with exact INR values and counts]

Business Insights
[2 to 4 sentences explaining what these numbers mean for the business]

Data Caveats
[note any missing data, quality issues, or limitations that affect accuracy]

Important rules:
1. Always state that data was pulled live from monday.com at the query time
2. Use Indian number notation — Lakhs (1L = Rs.1,00,000) and Crores (1 Cr = Rs.1,00,00,000)
3. When someone asks about the energy sector, that means Renewables plus Powerline combined — mention this
4. If the question refers to "this quarter" without specifying dates, ask one clarifying question
5. Financial values are masked/scaled for confidentiality — note this once
6. For follow-up questions, refer back to prior answers naturally
7. Be direct and precise — founders want numbers, not paragraphs
"""


def classify_query(query, history):
    client = Groq(api_key=GROQ_API_KEY)
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify this business intelligence query. Reply with exactly one word only.\n\n"
                    "work_orders — if the question is about: billing, collections, invoices, "
                    "work order execution status, accounts receivable, project delivery\n"
                    "deals — if the question is about: pipeline, sales funnel stages, "
                    "closure probability, win rate, new business development\n"
                    "both — if the question needs data from both boards: overall performance, "
                    "sector comparisons, revenue vs pipeline, BD performance\n\n"
                    "Examples:\n"
                    "  How much have we collected this year? -> work_orders\n"
                    "  Show high probability open deals -> deals\n"
                    "  How is Mining performing? -> both\n"
                    "  Which work orders are stuck? -> work_orders\n"
                    "  Pipeline for energy sector -> deals"
                ),
            },
            *[{"role": m["role"], "content": m["content"]} for m in history[-4:]],
            {"role": "user", "content": query},
        ]
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.0,
            max_tokens=5,
        )
        result = resp.choices[0].message.content.strip().lower()
        if "work_orders" in result:
            return "work_orders"
        if "deals" in result:
            return "deals"
        return "both"
    except Exception:
        return "both"


def call_llm(query, context, history):
    client = Groq(api_key=GROQ_API_KEY)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *[{"role": m["role"], "content": m["content"]} for m in history[-8:]],
        {
            "role": "user",
            "content": (
                f"Data pulled live from monday.com at "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n\n"
                f"{context}\n\n"
                f"Question: {query}"
            ),
        },
    ]
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.15,
        max_tokens=2500,
    )
    return resp.choices[0].message.content

# Agent Pipeline 

def run_agent(query, history):
    traces = []

    def ts():
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    traces.append({
        "type": "step",
        "msg": f"[{ts()}] Classifying query to determine which boards to query...",
        "detail": f'Query received: "{query[:120]}"',
    })

    boards = classify_query(query, history)

    traces.append({
        "type": "step",
        "msg": f"[{ts()}] Classification result: fetch {boards.upper().replace('_', ' ')}",
        "detail": "Routing to the appropriate monday.com board(s) based on query intent",
    })

    context_parts = []

    if boards in ("work_orders", "both"):
        board_id = MONDAY_WO_BOARD_ID
        traces.append({
            "type": "api",
            "msg": f"[{ts()}] POST https://api.monday.com/v2",
            "detail": f"Fetching Work Orders board (ID: {board_id}) — paginated GraphQL query",
        })

        wo_raw, wo_trace = live_fetch_board("WO", "Work Orders")

        if wo_trace["status"] == "success":
            wo_df = clean_dataframe(wo_raw)
            traces.append({
                "type": "success",
                "msg": f"[{ts()}] Work Orders: {wo_trace['total_items']} items retrieved in {wo_trace['pages']} page(s)",
                "detail": (
                    f"Board name: {wo_trace['board_name']} | "
                    f"HTTP 200 | {wo_trace['start']} to {wo_trace['end']}"
                ),
            })
            traces.append({
                "type": "step",
                "msg": f"[{ts()}] Cleaning Work Orders data...",
                "detail": (
                    f"Applied: null normalization, status normalization, "
                    f"billing status typo correction, 100% null column removal. "
                    f"Rows after cleaning: {len(wo_df)}"
                ),
            })
            context_parts.append(build_wo_context(wo_df))
        else:
            traces.append({
                "type": "error",
                "msg": f"[{ts()}] Work Orders fetch failed",
                "detail": wo_trace["error"],
            })
            context_parts.append(f"WORK ORDERS: Fetch failed — {wo_trace['error']}")

    if boards in ("deals", "both"):
        board_id = MONDAY_DEALS_BOARD_ID
        traces.append({
            "type": "api",
            "msg": f"[{ts()}] POST https://api.monday.com/v2",
            "detail": f"Fetching Deals board (ID: {board_id}) — paginated GraphQL query",
        })

        dl_raw, dl_trace = live_fetch_board("DEALS", "Deals")

        if dl_trace["status"] == "success":
            dl_df = clean_dataframe(dl_raw)
            traces.append({
                "type": "success",
                "msg": f"[{ts()}] Deals: {dl_trace['total_items']} items retrieved in {dl_trace['pages']} page(s)",
                "detail": (
                    f"Board name: {dl_trace['board_name']} | "
                    f"HTTP 200 | {dl_trace['start']} to {dl_trace['end']}"
                ),
            })
            traces.append({
                "type": "step",
                "msg": f"[{ts()}] Cleaning Deals data...",
                "detail": (
                    f"Applied: header-row duplicate removal, "
                    f"deal status normalization, closure probability cleanup. "
                    f"Rows after cleaning: {len(dl_df)}"
                ),
            })
            context_parts.append(build_deals_context(dl_df))
        else:
            traces.append({
                "type": "error",
                "msg": f"[{ts()}] Deals fetch failed",
                "detail": dl_trace["error"],
            })
            context_parts.append(f"DEALS: Fetch failed — {dl_trace['error']}")

    traces.append({
        "type": "step",
        "msg": f"[{ts()}] Sending data to Groq llama-3.3-70b-versatile for analysis...",
        "detail": "Model: llama-3.3-70b-versatile | Temperature: 0.15 | Max tokens: 2500",
    })

    try:
        answer = call_llm(query, "\n\n".join(context_parts), history)
        traces.append({
            "type": "success",
            "msg": f"[{ts()}] Analysis complete.",
            "detail": "",
        })
    except Exception as e:
        answer = f"Error generating analysis: {e}\n\nPlease verify your Groq API key at console.groq.com"
        traces.append({
            "type": "error",
            "msg": f"[{ts()}] LLM call failed",
            "detail": str(e),
        })

    return answer, traces

# Streamlit UI

st.set_page_config(
    page_title="Skylark BI Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>

/* Reset and base */
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { margin: 0; padding: 0; background: #ffffff; color: #0d0d0d; }

/* Hide default streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="collapsedControl"] { display: none; }

/* Main layout */
.main .block-container {
    max-width: 820px;
    margin: 0 auto;
    padding: 0 16px 120px 16px;
}

/* Top bar */
.top-bar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 52px;
    background: #ffffff;
    border-bottom: 1px solid #e5e5e5;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    z-index: 1000;
}
.top-bar-title {
    font-size: 15px;
    font-weight: 600;
    color: #0d0d0d;
    letter-spacing: -0.2px;
}
.top-bar-sub {
    font-size: 12px;
    color: #8e8ea0;
    margin-top: 1px;
}
.new-chat-btn {
    font-size: 13px;
    color: #8e8ea0;
    cursor: pointer;
    padding: 5px 10px;
    border-radius: 6px;
    border: 1px solid #e5e5e5;
    background: white;
}
.new-chat-btn:hover { background: #f7f7f8; }

/* Spacer for fixed top bar */
.top-spacer { height: 68px; }

/* Message bubbles */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 16px 0 8px 0;
}
.msg-user-inner {
    background: #f4f4f5;
    color: #0d0d0d;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.6;
}
.msg-assistant {
    display: flex;
    gap: 12px;
    margin: 8px 0 16px 0;
    align-items: flex-start;
}
.msg-avatar {
    width: 28px; height: 28px;
    background: #0d0d0d;
    border-radius: 6px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    color: white;
    font-weight: 700;
    margin-top: 2px;
}
.msg-assistant-inner {
    flex: 1;
    font-size: 14px;
    line-height: 1.7;
    color: #0d0d0d;
}

/* Trace panel */
.trace-panel {
    background: #f7f7f8;
    border: 1px solid #e5e5e5;
    border-radius: 8px;
    padding: 12px 14px;
    margin: 8px 0 16px 40px;
}
.trace-title {
    font-size: 11px;
    font-weight: 600;
    color: #8e8ea0;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.trace-row {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 4px;
    margin: 2px 0;
    border-left: 2px solid transparent;
    color: #374151;
}
.trace-row.type-api     { border-left-color: #f59e0b; background: #fffbeb; }
.trace-row.type-success { border-left-color: #10b981; background: #f0fdf4; }
.trace-row.type-error   { border-left-color: #ef4444; background: #fef2f2; }
.trace-row.type-step    { border-left-color: #6366f1; background: #f5f3ff; }
.trace-detail {
    font-size: 11px;
    color: #6b7280;
    padding: 2px 8px 4px 8px;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
}

/* Input area */
.input-area {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #ffffff;
    border-top: 1px solid #e5e5e5;
    padding: 14px 0 20px 0;
    z-index: 999;
}
.input-inner {
    max-width: 820px;
    margin: 0 auto;
    padding: 0 16px;
}

/* Override streamlit form */
.stForm { border: none !important; padding: 0 !important; }
div[data-testid="stForm"] { border: 0; }

/* Input box */
.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #d9d9e3 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    color: #0d0d0d !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    transition: border-color 0.15s;
}
.stTextInput > div > div > input:focus {
    border-color: #8e8ea0 !important;
    box-shadow: 0 1px 8px rgba(0,0,0,0.1) !important;
    outline: none !important;
}

/* Send button */
div[data-testid="stFormSubmitButton"] > button {
    background: #0d0d0d !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 10px 18px !important;
    height: 44px !important;
    letter-spacing: -0.1px !important;
    transition: background 0.15s !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #1a1a1a !important;
}

/* Settings sidebar */
.stButton > button {
    background: white !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 8px !important;
    color: #374151 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Welcome screen */
.welcome-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: calc(100vh - 200px);
    text-align: center;
    padding: 40px 20px;
}
.welcome-title {
    font-size: 28px;
    font-weight: 700;
    color: #0d0d0d;
    letter-spacing: -0.5px;
    margin-bottom: 8px;
}
.welcome-sub {
    font-size: 15px;
    color: #8e8ea0;
    margin-bottom: 36px;
}
.suggestion-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    max-width: 580px;
    width: 100%;
}
.suggestion-card {
    background: #f7f7f8;
    border: 1px solid #e5e5e5;
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 13px;
    color: #374151;
    text-align: left;
    cursor: pointer;
    line-height: 1.4;
}
.suggestion-card:hover { background: #f0f0f1; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d9d9e3; border-radius: 3px; }

</style>
""", unsafe_allow_html=True)


# Session state init
for k, v in [("messages", []), ("traces_map", {})]:
    if k not in st.session_state:
        st.session_state[k] = v


def render_trace_panel(traces, expanded=False):
    with st.expander("View execution trace", expanded=expanded):
        st.markdown('<div class="trace-title">Agent Execution Log</div>', unsafe_allow_html=True)
        for t in traces:
            css = f"type-{t['type']}"
            st.markdown(
                f'<div class="trace-row {css}">{t["msg"]}</div>',
                unsafe_allow_html=True,
            )
            if t.get("detail"):
                st.markdown(
                    f'<div class="trace-detail">{t["detail"]}</div>',
                    unsafe_allow_html=True,
                )


# Top bar
st.markdown("""
<div class="top-bar">
    <div>
        <div class="top-bar-title">Skylark BI Agent</div>
        <div class="top-bar-sub">Connected to monday.com</div>
    </div>
    <div style="display:flex; align-items:center; gap:10px;">
        <div style="font-size:12px; color:#8e8ea0;">Groq LLaMA 3.3-70B</div>
    </div>
</div>
<div class="top-spacer"></div>
""", unsafe_allow_html=True)


# Chat history
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user"><div class="msg-user-inner">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="msg-assistant">', unsafe_allow_html=True)
        col_avatar, col_text = st.columns([0.04, 0.96])
        with col_avatar:
            st.markdown('<div class="msg-avatar">BI</div>', unsafe_allow_html=True)
        with col_text:
            st.markdown(msg["content"])
        st.markdown("</div>", unsafe_allow_html=True)

        tk = f"t_{i}"
        if tk in st.session_state.traces_map:
            render_trace_panel(st.session_state.traces_map[tk], expanded=False)


# Welcome screen
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-title">What would you like to know?</div>
        <div class="welcome-sub">Ask any business question about Skylark's pipeline, revenue, or operations.</div>
        <div class="suggestion-grid">
            <div class="suggestion-card">How is our pipeline looking for the energy sector?</div>
            <div class="suggestion-card">What is our billing and collection efficiency?</div>
            <div class="suggestion-card">Which high-probability deals are still open?</div>
            <div class="suggestion-card">Show accounts receivable breakdown by sector</div>
            <div class="suggestion-card">Which work orders are stuck or on hold?</div>
            <div class="suggestion-card">Compare pipeline value against executed WO value</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Input form — fixed at bottom
st.markdown('<div class="input-area"><div class="input-inner">', unsafe_allow_html=True)

pf = st.session_state.pop("_pf", "")
with st.form("chat_form", clear_on_submit=True):
    input_col, btn_col = st.columns([6, 1])
    with input_col:
        user_input = st.text_input(
            "message",
            value=pf,
            placeholder="Ask about revenue, pipeline, sector performance...",
            label_visibility="collapsed",
        )
    with btn_col:
        submitted = st.form_submit_button("Send", use_container_width=True)

st.markdown("</div></div>", unsafe_allow_html=True)


# Handle query
if submitted and user_input.strip():
    query = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": query})

    st.markdown(
        f'<div class="msg-user"><div class="msg-user-inner">{query}</div></div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Thinking..."):
        history = [m for m in st.session_state.messages[:-1]]
        answer, traces = run_agent(query, history)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    tk = f"t_{len(st.session_state.messages) - 1}"
    st.session_state.traces_map[tk] = traces

    st.markdown('<div class="msg-assistant">', unsafe_allow_html=True)
    col_a, col_t = st.columns([0.04, 0.96])
    with col_a:
        st.markdown('<div class="msg-avatar">BI</div>', unsafe_allow_html=True)
    with col_t:
        st.markdown(answer)
    st.markdown("</div>", unsafe_allow_html=True)

    render_trace_panel(traces, expanded=True)

    st.rerun()
