
import os
import math
import sqlite3
import re
from datetime import datetime
from typing import Dict, Tuple, Optional

import pandas as pd
import streamlit as st

APP_TITLE = "Refurbished Phone Selling (Demo)"
DB_PATH = os.path.join(os.path.dirname(__file__), "inventory.db")

# -----------------------------
# Auth (Mock)
# -----------------------------
def check_password():
    """Simple mock authentication: set env var ADMIN_PASSWORD or default to 'admin'."""
    def login_form():
        st.text_input("Password", type="password", key="password")
        st.button("Log in", on_click=try_login)

    def try_login():
        pw = st.session_state.get("password", "")
        if pw == os.environ.get("ADMIN_PASSWORD", "admin"):
            st.session_state["auth_ok"] = True
        else:
            st.session_state["auth_ok"] = False

    if "auth_ok" not in st.session_state:
        st.session_state["auth_ok"] = False

    if not st.session_state["auth_ok"]:
        st.warning("Admin login required (demo). Default password is 'admin' or set ADMIN_PASSWORD env var.")
        login_form()
        st.stop()

# -----------------------------
# DB Helpers
# -----------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS phones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        brand TEXT NOT NULL,
        model TEXT NOT NULL,
        storage TEXT,
        color TEXT,
        condition TEXT NOT NULL,
        base_price REAL NOT NULL CHECK(base_price >= 0),
        stock INTEGER NOT NULL CHECK(stock >= 0),
        tags TEXT DEFAULT '',
        discontinued INTEGER NOT NULL DEFAULT 0 CHECK(discontinued IN (0,1))
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS prices (
        phone_id INTEGER,
        platform TEXT CHECK(platform IN ('X','Y','Z')),
        override_price REAL,
        PRIMARY KEY(phone_id, platform),
        FOREIGN KEY(phone_id) REFERENCES phones(id) ON DELETE CASCADE
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS listings (
        phone_id INTEGER,
        platform TEXT CHECK(platform IN ('X','Y','Z')),
        listed INTEGER NOT NULL DEFAULT 0 CHECK(listed IN (0,1)),
        last_result TEXT,
        updated_at TEXT,
        PRIMARY KEY(phone_id, platform),
        FOREIGN KEY(phone_id) REFERENCES phones(id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()

# -----------------------------
# Business Logic: Fees & Mapping
# -----------------------------
PLATFORM_FEES = {
    "X": {"percent": 0.10, "flat": 0.0},        # 10% fee
    "Y": {"percent": 0.08, "flat": 2.0},         # 8% + $2
    "Z": {"percent": 0.12, "flat": 0.0},         # 12%
}

CONDITION_MAP = {
    # Internal -> platform category
    "X": {"New": "New", "Good": "Good", "Scrap": "Scrap", "Usable": "Scrap"},
    "Y": {"New": "3 stars (Excellent)", "Good": "2 stars (Good)", "Scrap": "1 star (Usable)", "Usable": "1 star (Usable)"},
    "Z": {"New": "New", "Good": "Good", "Scrap": "Good", "Usable": "As New"},
}
SUPPORTED_CONDITIONS = {"New", "Good", "Scrap", "Usable"}

def sanitize_text(s: str, max_len: int = 120) -> str:
    s = (s or "").strip()
    s = re.sub(r"[\x00-\x1F\x7F]", " ", s)
    return s[:max_len]

def compute_listing_price(base_price: float, platform: str) -> float:
    """Compute price so net >= base_price after fees."""
    fee = PLATFORM_FEES[platform]
    pct, flat = fee["percent"], fee["flat"]
    price = (base_price + flat) / (1 - pct) if (1 - pct) > 0 else base_price + flat
    return round(price + 0.01, 2)

def compute_profit(price: float, base_price: float, platform: str) -> float:
    fee = PLATFORM_FEES[platform]
    pct, flat = fee["percent"], fee["flat"]
    net = price * (1 - pct) - flat
    return round(net - base_price, 2)

def platform_condition(internal_condition: str, platform: str) -> Optional[str]:
    mapping = CONDITION_MAP.get(platform, {})
    return mapping.get(internal_condition)

def can_list(phone: dict, platform: str, price: float) -> tuple[bool, str]:
    """Business rules to decide listing eligibility."""
    if phone["discontinued"] == 1 or "discontinued" in (phone["tags"] or ""):
        return False, "Discontinued product"
    if phone["stock"] <= 0 or "out_of_stock" in (phone["tags"] or ""):
        return False, "Out of stock"
    mapped = platform_condition(phone["condition"], platform)
    if not mapped:
        return False, "Unsupported condition for platform"
    profit = compute_profit(price, phone["base_price"], platform)
    if profit < 0:
        return False, f"Unprofitable (profit ${profit}) due to fees"
    return True, "Eligible"

# -----------------------------
# Data Access
# -----------------------------
def upsert_price_override(conn, phone_id: int, platform: str, override_price: Optional[float]):
    conn.execute(
        "INSERT INTO prices (phone_id, platform, override_price) VALUES (?,?,?) "
        "ON CONFLICT(phone_id, platform) DO UPDATE SET override_price=excluded.override_price;",
        (phone_id, platform, override_price),
    )

def get_effective_price(base_price: float, phone_id: int, platform: str, conn) -> float:
    cur = conn.execute("SELECT override_price FROM prices WHERE phone_id=? AND platform=?", (phone_id, platform))
    row = cur.fetchone()
    if row and row[0] is not None and row[0] > 0:
        return round(float(row[0]), 2)
    return compute_listing_price(base_price, platform)

def insert_phone(conn, phone: dict) -> int:
    cur = conn.execute("""
        INSERT INTO phones (brand, model, storage, color, condition, base_price, stock, tags, discontinued)
        VALUES (?,?,?,?,?,?,?,?,?);
    """, (
        sanitize_text(phone.get("brand")),
        sanitize_text(phone.get("model")),
        sanitize_text(phone.get("storage")),
        sanitize_text(phone.get("color")),
        sanitize_text(phone.get("condition")),
        float(phone.get("base_price", 0)),
        int(phone.get("stock", 0)),
        sanitize_text(phone.get("tags", ""), 300),
        int(bool(phone.get("discontinued", 0))),
    ))
    return cur.lastrowid

def update_phone(conn, phone_id: int, phone: dict):
    conn.execute("""
        UPDATE phones SET brand=?, model=?, storage=?, color=?, condition=?, base_price=?, stock=?, tags=?, discontinued=?
        WHERE id=?;
    """, (
        sanitize_text(phone.get("brand")),
        sanitize_text(phone.get("model")),
        sanitize_text(phone.get("storage")),
        sanitize_text(phone.get("color")),
        sanitize_text(phone.get("condition")),
        float(phone.get("base_price", 0)),
        int(phone.get("stock", 0)),
        sanitize_text(phone.get("tags", ""), 300),
        int(bool(phone.get("discontinued", 0))),
        phone_id
    ))

def delete_phone(conn, phone_id: int):
    conn.execute("DELETE FROM phones WHERE id=?;", (phone_id,))

def read_phones_df(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM phones ORDER BY id DESC;", conn)

def upsert_listing(conn, phone_id: int, platform: str, listed: bool, last_result: str):
    conn.execute(
        "INSERT INTO listings (phone_id, platform, listed, last_result, updated_at) VALUES (?,?,?,?,?) "
        "ON CONFLICT(phone_id, platform) DO UPDATE SET listed=excluded.listed, last_result=excluded.last_result, updated_at=excluded.updated_at;",
        (phone_id, platform, int(listed), last_result, datetime.utcnow().isoformat())
    )

def read_listings_df(conn) -> pd.DataFrame:
    return pd.read_sql_query("""
        SELECT l.phone_id, p.brand, p.model, p.storage, p.color, p.condition, l.platform, l.listed, l.last_result, l.updated_at
        FROM listings l
        JOIN phones p ON p.id = l.phone_id
        ORDER BY updated_at DESC;
    """, conn)

# -----------------------------
# UI Components
# -----------------------------
def render_header():
    st.title(APP_TITLE)
    st.caption("All features are simulated. No real platform integrations.")

def render_inventory_tab(conn):
    st.subheader("Inventory Management")
    df = read_phones_df(conn)
    st.dataframe(df, use_container_width=True, height=260)

    with st.expander("➕ Add / Update Phone"):
        col1, col2 = st.columns(2)
        with col1:
            phone_id = st.number_input("Phone ID (leave 0 to add new)", min_value=0, step=1, value=0)
            brand = st.text_input("Brand")
            model = st.text_input("Model")
            storage = st.text_input("Storage (e.g., 128GB)")
            color = st.text_input("Color")
        with col2:
            condition = st.selectbox("Condition", sorted(list(SUPPORTED_CONDITIONS)))
            base_price = st.number_input("Base Price (your minimum revenue)", min_value=0.0, step=1.0, value=0.0)
            stock = st.number_input("Stock", min_value=0, step=1, value=0)
            tags = st.text_input("Tags (comma separated, e.g., out_of_stock,clearance)")
            discontinued = st.checkbox("Discontinued?")

        if st.button("Save Phone"):
            try:
                assert brand and model and condition in SUPPORTED_CONDITIONS, "Invalid inputs"
                payload = {
                    "brand": brand, "model": model, "storage": storage, "color": color,
                    "condition": condition, "base_price": base_price, "stock": stock,
                    "tags": tags, "discontinued": 1 if discontinued else 0,
                }
                with conn:
                    if phone_id and (not df.empty and phone_id in set(df["id"].tolist())):
                        update_phone(conn, int(phone_id), payload)
                        st.success(f"Updated phone #{phone_id}")
                    else:
                        new_id = insert_phone(conn, payload)
                        st.success(f"Inserted phone #{new_id}")
            except Exception as e:
                st.error(f"Failed to save: {e}")

    with st.expander("🗑️ Delete Phone"):
        del_id = st.number_input("Phone ID to delete", min_value=0, step=1, value=0)
        if st.button("Delete"):
            try:
                with conn:
                    delete_phone(conn, int(del_id))
                st.success(f"Deleted phone #{del_id} (if existed)")
            except Exception as e:
                st.error(f"Delete failed: {e}")

def render_bulk_tab(conn):
    st.subheader("Bulk Upload (CSV)")
    st.caption("Headers required: brand, model, storage, color, condition, base_price, stock, tags, discontinued")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            required = {"brand","model","condition","base_price","stock"}
            assert required.issubset(df.columns), f"Missing columns: {sorted(list(required - set(df.columns)))}"
            inserted, updated, skipped = 0, 0, 0
            existing = read_phones_df(conn)
            existing_key = set((r.brand, r.model, r.storage, r.color) for r in existing.itertuples()) if not existing.empty else set()
            with conn:
                for r in df.itertuples():
                    if r.condition not in SUPPORTED_CONDITIONS:
                        skipped += 1
                        continue
                    payload = {
                        "brand": sanitize_text(r.brand),
                        "model": sanitize_text(r.model),
                        "storage": sanitize_text(getattr(r, "storage", "")),
                        "color": sanitize_text(getattr(r, "color", "")),
                        "condition": sanitize_text(r.condition),
                        "base_price": float(r.base_price),
                        "stock": int(r.stock),
                        "tags": sanitize_text(getattr(r, "tags", ""), 300),
                        "discontinued": int(getattr(r, "discontinued", 0)) if str(getattr(r, "discontinued", 0)).isdigit() else 0
                    }
                    key = (payload["brand"], payload["model"], payload["storage"], payload["color"])
                    if key in existing_key:
                        match_id = int(existing.query("brand == @key[0] and model == @key[1] and storage == @key[2] and color == @key[3]").iloc[0]["id"])
                        update_phone(conn, match_id, payload)
                        updated += 1
                    else:
                        insert_phone(conn, payload)
                        inserted += 1
            st.success(f"Bulk processed. Inserted: {inserted}, Updated: {updated}, Skipped: {skipped}")
        except Exception as e:
            st.error(f"Bulk upload failed: {e}")

    st.info("Need a template? Use the sample CSV that came with this project.")

def render_pricing_tab(conn):
    st.subheader("Automated Price Updates + Overrides")
    df = read_phones_df(conn)
    if df.empty:
        st.info("No phones yet.")
        return

    phone_id = st.selectbox("Select phone", df["id"].tolist(), format_func=lambda pid: f"#{pid} - {df[df.id==pid].iloc[0]['brand']} {df[df.id==pid].iloc[0]['model']}")
    phone = df[df.id == phone_id].iloc[0].to_dict()

    cols = st.columns(3)
    for i, platform in enumerate(["X","Y","Z"]):
        with cols[i]:
            base_price = float(phone["base_price"])
            effective = get_effective_price(base_price, int(phone["id"]), platform, conn)
            auto_price = compute_listing_price(base_price, platform)
            st.metric(f"Platform {platform}", f"${effective}", f"Auto: ${auto_price}")
            new_override = st.number_input(f"Override price for {platform} (0 to clear)", min_value=0.0, step=1.0, value=float(effective))
            if st.button(f"Save Override - {platform}"):
                with conn:
                    upsert_price_override(conn, int(phone["id"]), platform, None if new_override <= 0 else float(new_override))
                st.success(f"Saved override for phone #{phone['id']} on {platform}")

def render_listings_tab(conn):
    st.subheader("Mock Platform Listing")
    df = read_phones_df(conn)
    if df.empty:
        st.info("No phones yet.")
        return

    target_platform = st.selectbox("Select platform", ["X","Y","Z"])
    target_ids = st.multiselect("Select phones to list", df["id"].tolist(),
                                format_func=lambda pid: f"#{pid} - {df[df.id==pid].iloc[0]['brand']} {df[df.id==pid].iloc[0]['model']}")

    if st.button("Simulate Listing"):
        if not target_ids:
            st.warning("Pick at least one phone.")
        else:
            with conn:
                for pid in target_ids:
                    phone = df[df.id == pid].iloc[0].to_dict()
                    price = get_effective_price(float(phone["base_price"]), int(phone["id"]), target_platform, conn)
                    ok, reason = can_list(phone, target_platform, price)
                    upsert_listing(conn, int(pid), target_platform, ok, f"Price ${price} - {reason}")
            st.success("Simulation complete. See results below.")

    st.markdown("---")
    st.write("Latest Listings Results")
    ldf = read_listings_df(conn)
    if ldf.empty:
        st.info("No listing attempts yet.")
    else:
        st.dataframe(ldf, use_container_width=True, height=300)

def render_search_tab(conn):
    st.subheader("Search & Filter Inventory")
    df = read_phones_df(conn)
    if df.empty:
        st.info("No phones yet.")
        return

    q = st.text_input("Search by brand or model")
    cond = st.multiselect("Filter by condition", sorted(list(SUPPORTED_CONDITIONS)))
    platform = st.selectbox("Filter by platform listed", ["Any","X","Y","Z"])

    filtered = df.copy()
    if q:
        ql = q.lower()
        filtered = filtered[filtered["brand"].str.lower().str.contains(ql) | filtered["model"].str.lower().str.contains(ql)]
    if cond:
        filtered = filtered[filtered["condition"].isin(cond)]

    if platform != "Any":
        ldf = read_listings_df(conn)
        if ldf.empty:
            filtered = filtered.iloc[0:0]
        else:
            listed_ids = set(ldf[(ldf.platform==platform) & (ldf.listed==1)]["phone_id"].tolist())
            filtered = filtered[filtered["id"].isin(listed_ids)]

    st.dataframe(filtered, use_container_width=True, height=320)

def render_export_tab(conn):
    st.subheader("Export Inventory CSV")
    df = read_phones_df(conn)
    if df.empty:
        st.info("No phones to export.")
        return
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="inventory_export.csv", mime="text/csv")

# -----------------------------
# Main App
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    check_password()
    init_db()
    render_header()

    tabs = st.tabs(["Inventory", "Bulk Upload", "Pricing", "Listings", "Search/Filter", "Export"])
    with tabs[0]:
        conn = get_conn()
        render_inventory_tab(conn)
        conn.close()

    with tabs[1]:
        conn = get_conn()
        render_bulk_tab(conn)
        conn.close()

    with tabs[2]:
        conn = get_conn()
        render_pricing_tab(conn)
        conn.close()

    with tabs[3]:
        conn = get_conn()
        render_listings_tab(conn)
        conn.close()

    with tabs[4]:
        conn = get_conn()
        render_search_tab(conn)
        conn.close()

    with tabs[5]:
        conn = get_conn()
        render_export_tab(conn)
        conn.close()

if __name__ == "__main__":
    main()
