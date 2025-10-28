# app.py
"""
Interactive Net Worth Forecaster with Generic Accounts + Monte Carlo + ISA global allowance

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from typing import Dict, Optional
from datetime import datetime

from data.account import Account
from tax.uk.tax import uk_income_tax_2024


# -----------------------
# Core model
# -----------------------


def simulate(cfg: Dict, randomize=False, rng: Optional[np.random.Generator]=None) -> pd.DataFrame:
    ISA_ALLOWANCE = 20000.0

    years = list(range(cfg["start_year"], cfg["start_year"] + (cfg["end_age"] - cfg["current_age"]) + 1))
    ages = list(range(cfg["current_age"], cfg["end_age"] + 1))

    accounts = {acc["name"]: Account(**acc) for acc in cfg["accounts"]}

    base_salary = cfg["base_salary"]
    bonus = cfg["bonus"]

    rows = []
    for i, (year, age) in enumerate(zip(years, ages)):
        is_retired = age >= cfg["retirement_age"]

        # --- ISA allowance tracker (resets each year) ---
        isa_contrib_used = 0.0

        # --- Contributions ---
        gross_income_for_tax = 0.0 if is_retired else base_salary + bonus
        pre_tax_total = 0.0
        for acc in accounts.values():
            if not is_retired and acc.pre_tax_contrib_rate > 0:
                contrib = base_salary * acc.pre_tax_contrib_rate
                acc.balance += contrib
                pre_tax_total += contrib
        gross_income_for_tax -= pre_tax_total

        # Tax
        tax = 0.0 if is_retired else uk_income_tax_2024(gross_income_for_tax)
        net_income = gross_income_for_tax - tax

        # Post-tax contributions + income sink
        for acc in accounts.values():
            if acc.is_income_sink:
                acc.balance += net_income
            if not is_retired and acc.post_tax_contrib_rate > 0:
                contrib = net_income * acc.post_tax_contrib_rate
                if acc.is_isa:
                    allowed = max(0, ISA_ALLOWANCE - isa_contrib_used)
                    contrib = min(contrib, allowed)
                    isa_contrib_used += contrib
                acc.balance += contrib
                net_income -= contrib

        # --- Expenses ---
        expenses = (cfg["expenses_retired"] if is_retired else cfg["expenses_working"]) * ((1 + cfg["inflation_rate"])**i)
        remaining = expenses
        for acc_name in cfg["drawdown_order"]:
            acc = accounts[acc_name]
            if remaining <= 0:
                break
            if not acc.is_expense_source:
                continue
            draw = min(acc.balance, remaining)
            acc.balance -= draw
            remaining -= draw
        # if remaining > 0: expenses not fully covered (negative drift)

        # --- Investment growth ---
        for acc in accounts.values():
            gross = 1 + acc.exp_return
            if randomize and acc.vol > 0 and rng is not None:
                gross += rng.normal(0.0, acc.vol)
            acc.balance *= gross

        rows.append({
            "year": year,
            "age": age,
            **{f"bal_{k}": v.balance for k,v in accounts.items()},
            "net_worth": sum(v.balance for v in accounts.values())
        })

        # --- Growth of salary/bonus ---
        base_salary *= (1 + cfg["salary_growth"])
        bonus *= (1 + cfg["bonus_growth"])

    return pd.DataFrame(rows)

def monte_carlo(cfg: Dict, runs: int=200, seed: int=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    all_runs = []
    for i in range(runs):
        df = simulate(cfg, randomize=True, rng=rng)
        df["run"] = i
        all_runs.append(df[["year","net_worth","run"]])
    mc = pd.concat(all_runs)
    return mc.groupby("year")["net_worth"].quantile([0.1,0.5,0.9]).unstack()

if __name__ == "__main__":
    # -----------------------
    # Streamlit UI
    # -----------------------

    st.title("💷 Interactive Net Worth Forecaster (not to be taken as financial advice)")

    # General params
    with st.sidebar:
        st.header("Parameters")
        current_age = st.number_input("Current age", 18, 80, 25)
        retirement_age = st.number_input("Retirement age", 40, 75, 65)
        end_age = st.number_input("End age", 60, 110, 90)
        inflation_rate = st.slider("Inflation rate", 0.0, 10.0, 2.5, 0.1)
        base_salary = st.number_input("Base salary (£)", 0, 1_000_000, 0, 1000)
        bonus = st.number_input("Bonus (£)", 0, 1_000_000, 0, 1000)
        salary_growth = st.slider("Salary growth rate %", 0.0, 10.0, 3.0, 0.5)
        bonus_growth = st.slider("Bonus growth rate %", -10.0, 10.0, 0.0, 0.5)

        st.subheader("Expenses")
        exp_working = st.number_input("Annual expenses (working)", 0, 2000000, 0, 1000)
        exp_retired = st.number_input("Annual expenses (retired)", 0, 2000000, 0, 1000)

    # Accounts
    st.header("Accounts")
    accounts = []
    n_acc = st.number_input("Number of accounts", 1, 100, 1)
    for i in range(n_acc):
        with st.expander(f"Account {i+1}"):
            name = st.text_input(f"Name {i+1}", value=f"Account", key=f"name_{i}")
            bal = st.number_input(f"Starting balance ", 0, 10_000_000, 1_000, 1000, key=f"bal_{i}")
            exp_ret = st.slider(f"Expected return %", -10.0, 20.0, 5.0, 0.5, key=f"exp_ret_{i}")
            vol = st.slider(f"Volatility %", 0.0, 50.0, 10.0, 0.5, key=f"vol_{i}")
            pre_tax_rate = st.slider(f"Pre-tax contrib rate % of salary", 0.0, 50.0, 0.0, 1.0, key=f"pre_tax_rate_{i}")
            post_tax_rate = st.slider(f"Post-tax contrib rate % of salary", 0.0, 50.0, 0.0, 1.0, key=f"post_tax_rate_{i}")
            is_income_sink = st.checkbox(f"Income sink?", value=(i==0), key=f"income_sink_{i}")
            is_expense_source = st.checkbox(f"Expense source?", value=True, key=f"expense_source_{i}")
            is_isa = st.checkbox(f"ISA account?", value=False, key=f"isa_{i}")

            accounts.append({
                "name": name,
                "balance": bal,
                "exp_return": exp_ret / 100,
                "vol": vol / 100,
                "pre_tax_contrib_rate": pre_tax_rate / 100,
                "post_tax_contrib_rate": post_tax_rate / 100,
                "is_income_sink": is_income_sink,
                "is_expense_source": is_expense_source,
                "is_isa": is_isa
            })

    # Drawdown order
    drawdown_order = st.multiselect("Drawdown order", [a["name"] for a in accounts], default=[a["name"] for a in accounts])

    # Monte Carlo controls
    st.sidebar.subheader("Monte Carlo")
    do_mc = st.sidebar.checkbox("Enable Monte Carlo?", value=True)
    n_runs = st.sidebar.slider("Number of runs", 50, 1000, 200, 50)

    # Build config
    cfg = {
        "start_year": datetime.now().year,
        "current_age": current_age,
        "retirement_age": retirement_age,
        "end_age": end_age,
        "inflation_rate": inflation_rate / 100,
        "base_salary": base_salary,
        "bonus": bonus,
        "salary_growth": salary_growth / 100,
        "bonus_growth": bonus_growth / 100,
        "expenses_working": exp_working,
        "expenses_retired": exp_retired,
        "accounts": accounts,
        "drawdown_order": drawdown_order
    }

    # Deterministic simulation
    df = simulate(cfg)

    # Monte Carlo
    mc = None
    if do_mc:
        mc = monte_carlo(cfg, runs=n_runs)

    # Plot
    st.subheader("Forecast")
    fig, ax = plt.subplots(figsize=(10,6))
    for acc in accounts:
        ax.plot(df["year"], df[f"bal_{acc['name']}"], label=acc["name"])
    ax.plot(df["year"], df["net_worth"], label="Net Worth (det)", color="k", linewidth=2, linestyle="--")

    if mc is not None:
        ax.fill_between(mc.index, mc[0.1], mc[0.9], color="blue", alpha=0.2, label="MC 10-90%")
        ax.plot(mc.index, mc[0.5], color="blue", linestyle=":", linewidth=2, label="MC Median")

    ax.legend()
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("£{x:,.0f}"))
    ax.set_ylabel("£ (nominal)")
    ax.set_xlabel("Year")
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)

    # Table
    st.subheader("Results table")
    st.dataframe(df)