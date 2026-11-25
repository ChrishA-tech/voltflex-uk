"""Streamlit Live Grid & EV Dispatch Command Center for VoltFlex UK."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from src.agents.flexibility_copilot import FlexibilityCopilot
from src.analytics.curtailment_detector import (
    build_curtailment_summary,
    detect_grid_regimes,
    summarize_dno_headroom,
)
from src.ingestion.grid_telemetry import DEFAULT_DB_PATH, UK_DNO_REGIONS
from src.models.forecast_and_dispatch import (
    DispatchConstraints,
    train_and_evaluate_rl_dispatcher,
)

st.set_page_config(page_title="VoltFlex UK — Grid & EV Dispatch Command Center", layout="wide")


@st.cache_data(ttl=60)
def load_telemetry(db_path: str) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()
    with duckdb.connect(str(path), read_only=True) as con:
        df = con.execute("SELECT * FROM grid_telemetry ORDER BY timestamp").fetchdf()
    return detect_grid_regimes(df) if not df.empty else df


def main() -> None:
    st.title("VoltFlex UK — Real-Time Grid Balancing & EV Dispatch Command Center")
    st.caption("14 UK DNO Regions | LightGBM + MAPIE Conformal Forecasts | G98/G99 RL Shield | NESO/Ofgem Copilot")

    df = load_telemetry(str(DEFAULT_DB_PATH))
    if df.empty:
        st.warning("No telemetry found in DuckDB yet. Run `python -m src.run_pipeline` to populate data.")
        return

    summary = build_curtailment_summary(df)
    dno_summary = summarize_dno_headroom(df)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Validated Half-Hours", f"{summary.total_intervals:,}")
    k2.metric("Wind Curtailment Windows", f"{summary.wind_curtailment_windows:,}")
    k3.metric("Negative Agile Plunges", f"{summary.negative_agile_windows:,}")
    k4.metric("Est. Curtailed Wind", f"{summary.total_estimated_curtailed_mwh:,.1f} MWh")

    st.sidebar.header("Site & DNO Constraints")
    selected_region = st.sidebar.selectbox(
        "DNO Region",
        options=sorted(UK_DNO_REGIONS.keys()),
        format_func=lambda rid: f"{rid} — {UK_DNO_REGIONS[rid]['name']}",
    )
    g99_approved = st.sidebar.checkbox("G99 DNO Approval (> 3.68 kW export)", value=False)
    thermal_amps = st.sidebar.slider("Thermal Main Fuse Limit (A @ 230V)", 32, 100, 60)

    tab_grid, tab_rl, tab_copilot = st.tabs(
        ["14-Region DNO Headroom & Plunge Regimes", "RL EV/Battery Dispatch", "NESO/Ofgem Flexibility Copilot"]
    )

    with tab_grid:
        st.subheader("DNO Substation Transformer Headroom Across All 14 UK Regions")
        st.dataframe(dno_summary, use_container_width=True)

        reg_df = df[df["dno_region_id"] == selected_region].sort_values("timestamp")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Agile Import Price (p/kWh) — {UK_DNO_REGIONS[selected_region]['name']}**")
            st.line_chart(reg_df.set_index("timestamp")[["agile_import_pence"]])
        with c2:
            st.markdown(f"**Carbon Intensity (gCO2/kWh) — {UK_DNO_REGIONS[selected_region]['name']}**")
            st.line_chart(reg_df.set_index("timestamp")[["carbon_intensity_gco2_kwh"]])

    with tab_rl:
        st.subheader("PyTorch Policy-Gradient EV & Battery Dispatch (G98/G99 Shielded)")
        reg_df = df[df["dno_region_id"] == selected_region].sort_values("timestamp")
        constraints = DispatchConstraints(
            g99_approved=g99_approved,
            thermal_current_limit_amps=float(thermal_amps),
        )
        _, sim = train_and_evaluate_rl_dispatcher(reg_df, constraints=constraints, epochs=20)
        m1, m2, m3 = st.columns(3)
        m1.metric("RL Dispatch Cost", f"£{sim.total_cost_gbp:.2f}", f"Baseline £{sim.baseline_cost_gbp:.2f}")
        m2.metric("RL Net Carbon", f"{sim.total_carbon_kg:.1f} kg", f"Baseline {sim.baseline_carbon_kg:.1f} kg")
        m3.metric("Thermal / G98 Violation Rate", f"{sim.thermal_violation_rate * 100:.2f}%")

        sched = sim.schedule_df.set_index("timestamp")
        st.line_chart(sched[["ev_dispatch_kw", "battery_dispatch_kw", "net_site_kw"]])
        st.line_chart(sched[["ev_soc_pct", "battery_soc_pct"]])

    with tab_copilot:
        st.subheader("Guardrailed NESO DFS / Ofgem G98-G99 / SEG Copilot")
        user_q = st.text_input(
            "Ask a UK flexibility, G98/G99, DFS, or Smart Export Guarantee question:",
            value="Can I export 6.5 kW from my home battery during a NESO DFS event without G99 approval?",
        )
        export_kw = st.number_input("Requested Export Power (kW)", min_value=0.0, max_value=30.0, value=6.5, step=0.1)
        if st.button("Ask Copilot"):
            copilot = FlexibilityCopilot()
            res = copilot.ask(
                query=user_q,
                requested_export_kw=float(export_kw),
                g99_approved=g99_approved,
            )
            if res["interlock_triggered"]:
                st.error(res["interlock_reason"])
            st.markdown(res["answer"])
            st.caption(f"Verified Citations: {res['citations']} | Judge: {res['verification_notes']}")


if __name__ == "__main__":
    main()
