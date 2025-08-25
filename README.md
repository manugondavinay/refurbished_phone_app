# Refurbished Phone Selling (Demo)

A single-file **Streamlit** app with **SQLite** that demonstrates:
- Inventory management (CRUD, bulk CSV upload)
- Automated pricing per platform with fee structures (X, Y, Z) and manual overrides
- Condition mapping for each platform
- Mock platform listing with profitability and stock checks
- Search & filter, export
- Mock authentication (password: `admin` by default)

## Quick Start

```bash
cd refurbished_phone_app
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
export ADMIN_PASSWORD=admin   # optional, change if you like
streamlit run app.py
```

Open the provided local URL in your browser. Log in using the password you set (default `admin`).

## Bulk Upload Template
Use `sample_inventory.csv` to see the header format and example rows.

## Notes
- All integrations are simulated; no external APIs.
- Price calculation ensures the **net** (after platform fees) meets or exceeds your `base_price`.
  - X: 10% fee
  - Y: 8% fee + $2
  - Z: 12% fee
- Listing is blocked if out of stock, discontinued, unsupported condition, or unprofitable.
- This code is for demo/assignment purposes. For production, add proper auth, role-based access,
  input schemas, migrations, tests, etc.