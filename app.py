import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
from financial_model import FinancialModel

# Page configuration
st.set_page_config(
    page_title="Fintech Financial Model",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the financial model
@st.cache_resource
def get_model():
    return FinancialModel()

model = get_model()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #f9d923;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #23272f;
        color: #fff;
        padding: 1rem;
        border-radius: 0.7rem;
        border: 1.5px solid #444;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18);
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        color: #f9d923;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 {
        color: #fff;
        font-size: 2.1rem;
        font-weight: bold;
        margin-bottom: 0.2rem;
    }
    .metric-card p {
        color: #bbb;
        font-size: 1rem;
        margin-bottom: 0;
    }
    .metric-card .green {
        color: #4caf50 !important;
        font-weight: bold;
    }
    .metric-card .red {
        color: #e53935 !important;
        font-weight: bold;
    }
    .scenario-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def get_default_assumptions():
    """Get default assumptions for the model"""
    return {
        'start_date': date(2024, 1, 1),
        'starting_clients': 10,
        'monthly_client_growth_rate': 0.05,  # 5% monthly growth
        'client_churn_rate': 0.01,  # 1% monthly churn
        'starting_aua_per_client': 500000,  # $500k per client
        'aua_growth_rate': 0.02,  # 2% monthly AUA growth
        'aua_fee_bps': 25,  # 25 basis points
        'monthly_transactions_per_client': 5,
        'transaction_fee': 10,  # $10 per transaction
        'loan_frequency': 2,  # 2 loans per year per client
        'avg_loan_size': 100000,  # $100k average loan
        'lending_fee_bps': 100,  # 100 basis points
        'saas_monthly_fee': 500,  # $500 per month
        'saas_enabled': True,
        'fund_revenue_bps': 5,  # 5 basis points
        'monthly_fixed_costs': 50000,  # $50k monthly fixed costs
        'cost_per_client': 100,  # $100 per client per month
        'cost_per_mm_aua': 1000,  # $1k per $1MM AUA
        'starting_capital': 1000000,  # $1MM starting capital
        'projection_months': 60  # Default to 60 months
    }

def create_sidebar_inputs():
    """Create sidebar inputs for all assumptions"""
    st.sidebar.header("📊 Model Assumptions")
    
    # Load default assumptions
    if 'assumptions' not in st.session_state:
        st.session_state.assumptions = get_default_assumptions()
    
    assumptions = st.session_state.assumptions
    
    # Projection period selector
    projection_options = {
        'Just Now': 1,
        '12 Months': 12,
        '24 Months': 24,
        '48 Months': 48,
        '60 Months': 60
    }
    projection_label = st.sidebar.selectbox(
        "Projection Period",
        list(projection_options.keys()),
        index=4  # default to 60 Months
    )
    projection_months = projection_options[projection_label]
    st.session_state['projection_months'] = projection_months
    
    # Date and Capital
    st.sidebar.subheader("📅 Timeline & Capital")
    assumptions['start_date'] = st.sidebar.date_input(
        "Start Date", 
        value=assumptions['start_date']
    )
    assumptions['starting_capital'] = st.sidebar.number_input(
        "Business Starting Capital ($) (your company cash, not client money)", 
        value=float(assumptions['starting_capital']),
        step=100000.0,
        format="%.0f"
    )
    
    # Client Assumptions
    st.sidebar.subheader("👥 Client Growth")
    assumptions['starting_clients'] = st.sidebar.number_input(
        "Starting Clients", 
        value=int(assumptions['starting_clients']),
        min_value=1,
        step=1
    )
    assumptions['monthly_client_growth_rate'] = st.sidebar.slider(
        "Monthly Client Growth Rate (%)", 
        min_value=0.0, max_value=0.2, value=assumptions['monthly_client_growth_rate'],
        step=0.001, format="%.1f%%"
    )
    assumptions['client_churn_rate'] = st.sidebar.slider(
        "Monthly Client Churn Rate (%)", 
        min_value=0.0, max_value=0.1, value=assumptions['client_churn_rate'],
        step=0.001, format="%.1f%%"
    )
    
    # AUA Assumptions
    st.sidebar.subheader("💰 AUA (Assets Under Administration)")
    assumptions['starting_aua_per_client'] = st.sidebar.number_input(
        "Starting AUA per Client ($)", 
        value=float(assumptions['starting_aua_per_client']),
        step=10000.0,
        format="%.0f"
    )
    assumptions['aua_growth_rate'] = st.sidebar.slider(
        "Monthly AUA Growth Rate (%)", 
        min_value=0.0, max_value=0.1, value=assumptions['aua_growth_rate'],
        step=0.001, format="%.1f%%"
    )
    assumptions['aua_fee_bps'] = st.sidebar.slider(
        "AUA Fee (basis points)", 
        min_value=0, max_value=100, value=int(assumptions['aua_fee_bps']),
        step=1
    )
    
    # Transaction Revenue
    st.sidebar.subheader("💳 Transaction Revenue")
    assumptions['monthly_transactions_per_client'] = st.sidebar.number_input(
        "Monthly Transactions per Client", 
        value=float(assumptions['monthly_transactions_per_client']),
        min_value=0.0,
        step=0.5
    )
    assumptions['transaction_fee'] = st.sidebar.number_input(
        "Transaction Fee ($)", 
        value=float(assumptions['transaction_fee']),
        min_value=0.0,
        step=1.0
    )
    
    # Lending Revenue
    st.sidebar.subheader("🏦 Lending Revenue")
    assumptions['loan_frequency'] = st.sidebar.number_input(
        "Loans per Client per Year", 
        value=float(assumptions['loan_frequency']),
        min_value=0.0,
        step=0.1
    )
    assumptions['avg_loan_size'] = st.sidebar.number_input(
        "Average Loan Size ($)", 
        value=float(assumptions['avg_loan_size']),
        step=1000.0,
        format="%.0f"
    )
    assumptions['loan_duration_days'] = st.sidebar.number_input(
        "Average Loan Duration (days)",
        value=int(assumptions.get('loan_duration_days', 4)),
        min_value=1,
        max_value=7,
        step=1
    )
    assumptions['lending_fee_bps'] = st.sidebar.slider(
        "Lending Fee (basis points)", 
        min_value=0, max_value=500, value=int(assumptions['lending_fee_bps']),
        step=5
    )
    
    # SaaS Revenue
    st.sidebar.subheader("☁️ SaaS Revenue")
    assumptions['saas_enabled'] = st.sidebar.checkbox(
        "Enable SaaS Revenue", 
        value=assumptions['saas_enabled']
    )
    if assumptions['saas_enabled']:
        assumptions['saas_monthly_fee'] = st.sidebar.number_input(
            "SaaS Monthly Fee ($)", 
            value=float(assumptions['saas_monthly_fee']),
            min_value=0.0,
            step=10.0
        )
    
    # Fund Revenue
    st.sidebar.subheader("🏛️ Fund Revenue")
    assumptions['fund_revenue_bps'] = st.sidebar.slider(
        "Fund Revenue (basis points)", 
        min_value=0, max_value=50, value=int(assumptions['fund_revenue_bps']),
        step=1
    )
    
    # Cost Assumptions
    st.sidebar.subheader("💸 Cost Structure")
    assumptions['monthly_fixed_costs'] = st.sidebar.number_input(
        "Monthly Fixed Costs ($)", 
        value=float(assumptions['monthly_fixed_costs']),
        step=1000.0,
        format="%.0f"
    )
    assumptions['cost_per_client'] = st.sidebar.number_input(
        "Cost per Client per Month ($)", 
        value=float(assumptions['cost_per_client']),
        step=10.0
    )
    assumptions['cost_per_mm_aua'] = st.sidebar.number_input(
        "Cost per $1MM AUA ($)", 
        value=float(assumptions['cost_per_mm_aua']),
        step=100.0
    )
    
    assumptions['projection_months'] = projection_months
    
    # Remove the old months-based duration from assumptions if present
    if 'loan_duration_months' in assumptions:
        del assumptions['loan_duration_months']
    
    return assumptions

def display_dashboard(results, assumptions):
    """Display the main dashboard with key metrics"""
    st.markdown('<h1 class="main-header">💰 Fintech Financial Model</h1>', unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Current Revenue</h3>
            <h2>${results['total_revenue'].iloc[-1]:,.0f}</h2>
            <p>per month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total AUA</h3>
            <h2>${results['total_aua'].iloc[-1]:,.0f}</h2>
            <p>${results['total_aua'].iloc[-1]/results['total_clients'].iloc[-1]:,.0f} per client</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        breakeven_month = results['breakeven_month'].iloc[-1]
        if breakeven_month > 0:
            breakeven_text = f"Month {breakeven_month}"
        else:
            breakeven_text = "Never"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Breakeven</h3>
            <h2>{breakeven_text}</h2>
            <p>AUA needed: ${results['breakeven_aua'].iloc[-1]:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        current_cash_flow = results['cumulative_cash_flow'].iloc[-1]
        if current_cash_flow < 0:
            cash_flow_text = f"${abs(current_cash_flow):,.0f}"
            cash_flow_color = "red"
        else:
            cash_flow_text = f"${current_cash_flow:,.0f}"
            cash_flow_color = "green"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Cash Flow</h3>
            <h2 class=\"{'green' if current_cash_flow >= 0 else 'red'}\">{cash_flow_text}</h2>
            <p>cumulative</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

def display_charts(results, scenario_name="Base Case"):
    """Display interactive charts"""
    st.subheader("📈 Financial Charts")
    
    # Create charts
    charts = model.create_charts(results, scenario_name)
    
    # Display charts in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Cash Burn", "Revenue vs Costs", "Revenue Mix", "AUA Growth"])
    
    with tab1:
        st.plotly_chart(charts['cash_burn'], use_container_width=True)
    
    with tab2:
        st.plotly_chart(charts['revenue_costs'], use_container_width=True)
    
    with tab3:
        st.plotly_chart(charts['revenue_mix'], use_container_width=True)
    
    with tab4:
        st.plotly_chart(charts['aua_growth'], use_container_width=True)

def display_monthly_table(results):
    """Display monthly projections table"""
    st.subheader("📊 Monthly Projections")
    
    # Select columns to display
    display_columns = [
        'total_clients', 'total_aua', 'total_revenue', 'total_costs', 
        'net_income', 'cumulative_cash_flow'
    ]
    
    # Format the table
    display_df = results[display_columns].copy()
    display_df.columns = [
        'Clients', 'Total AUA', 'Revenue', 'Costs', 
        'Net Income', 'Cumulative Cash Flow'
    ]
    
    # Format numbers
    for col in ['Total AUA', 'Revenue', 'Costs', 'Net Income', 'Cumulative Cash Flow']:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
    
    display_df['Clients'] = display_df['Clients'].apply(lambda x: f"{x:.0f}")
    
    st.dataframe(display_df, use_container_width=True)

def run_monte_carlo_simulation(assumptions):
    """Run and display Monte Carlo simulation results"""
    st.subheader("🎲 Monte Carlo Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_simulations = st.number_input(
            "Number of Simulations", 
            min_value=100, max_value=10000, value=1000, step=100
        )
        
        if st.button("Run Monte Carlo Simulation", type="primary"):
            with st.spinner("Running Monte Carlo simulation..."):
                mc_results = model.run_monte_carlo(assumptions, n_simulations)
                st.session_state.mc_results = mc_results
    
    with col2:
        if 'mc_results' in st.session_state:
            mc_results = st.session_state.mc_results
            
            # Display key metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if mc_results['breakeven_months']:
                    avg_breakeven = np.mean(mc_results['breakeven_months'])
                    st.metric("Avg Breakeven (months)", f"{avg_breakeven:.1f}")
                else:
                    st.metric("Avg Breakeven (months)", "Never")
            
            with col_b:
                survival_pct = mc_results['survival_probability'] * 100
                st.metric("Survival Probability", f"{survival_pct:.1f}%")
            
            with col_c:
                avg_year1_revenue = np.mean(mc_results['year1_revenue'])
                st.metric("Avg Year 1 Revenue", f"${avg_year1_revenue:,.0f}")
    
    # Display comprehensive summary statistics
    if 'mc_results' in st.session_state:
        mc_results = st.session_state.mc_results
        
        st.subheader("📊 Monte Carlo Summary Statistics")
        st.write("Detailed statistics across all simulations:")
        
        # Calculate summary statistics
        summary_df = model.calculate_monte_carlo_summary_stats(mc_results)
        
        # Format the summary table for display
        display_summary = summary_df.copy()
        
        # Format currency columns
        currency_columns = ['Mean', 'Median', 'Min', 'Max', '10th Percentile', '25th Percentile', '75th Percentile', '90th Percentile', 'Std Dev']
        for col in currency_columns:
            if col in display_summary.columns:
                # Check if the metric is currency (contains $)
                is_currency = display_summary['Metric'].str.contains('\$').any()
                if is_currency:
                    display_summary[col] = display_summary[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                else:
                    display_summary[col] = display_summary[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        
        # Display the summary table
        st.dataframe(display_summary, use_container_width=True)
        
        # Add insights section
        st.subheader("💡 Key Insights")
        
        # Calculate some key insights
        if mc_results.get('final_cash_flow'):
            final_cf_array = np.array(mc_results['final_cash_flow'])
            positive_cf_prob = (final_cf_array > 0).mean() * 100
            negative_cf_prob = (final_cf_array < 0).mean() * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Probability of Positive Final Cash Flow", f"{positive_cf_prob:.1f}%")
            with col2:
                st.metric("Probability of Negative Final Cash Flow", f"{negative_cf_prob:.1f}%")
            with col3:
                if mc_results.get('breakeven_months'):
                    breakeven_array = np.array(mc_results['breakeven_months'])
                    early_breakeven_prob = (breakeven_array <= 24).mean() * 100
                    st.metric("Probability of Breakeven ≤ 24 months", f"{early_breakeven_prob:.1f}%")
        
        # Create Monte Carlo charts
        st.subheader("📈 Monte Carlo Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if mc_results['breakeven_months']:
                fig_breakeven = px.histogram(
                    x=mc_results['breakeven_months'],
                    title="Distribution of Breakeven Months",
                    labels={'x': 'Breakeven Month', 'y': 'Frequency'},
                    nbins=20
                )
                st.plotly_chart(fig_breakeven, use_container_width=True)
            
            # Final cash flow distribution
            if mc_results.get('final_cash_flow'):
                fig_cash_flow = px.histogram(
                    x=mc_results['final_cash_flow'],
                    title="Distribution of Final Cash Flow",
                    labels={'x': 'Final Cash Flow ($)', 'y': 'Frequency'},
                    nbins=30
                )
                fig_cash_flow.add_vline(x=0, line_dash="dash", line_color="red", 
                                       annotation_text="Breakeven Line")
                st.plotly_chart(fig_cash_flow, use_container_width=True)
        
        with col2:
            # Final AUA distribution
            if mc_results.get('final_aua'):
                fig_aua = px.histogram(
                    x=mc_results['final_aua'],
                    title="Distribution of Final AUA",
                    labels={'x': 'Final AUA ($)', 'y': 'Frequency'},
                    nbins=30
                )
                st.plotly_chart(fig_aua, use_container_width=True)
            
            # Revenue distributions
            fig_revenue = go.Figure()
            fig_revenue.add_trace(go.Box(
                y=mc_results['year1_revenue'],
                name='Year 1 Revenue',
                boxpoints='outliers'
            ))
            fig_revenue.add_trace(go.Box(
                y=mc_results['year2_revenue'],
                name='Year 2 Revenue',
                boxpoints='outliers'
            ))
            fig_revenue.add_trace(go.Box(
                y=mc_results['year3_revenue'],
                name='Year 3 Revenue',
                boxpoints='outliers'
            ))
            fig_revenue.update_layout(
                title="Revenue Distribution by Year",
                yaxis_title="Revenue ($)"
            )
            st.plotly_chart(fig_revenue, use_container_width=True)

def create_advanced_risk_analysis(assumptions):
    """Create advanced risk analysis with tornado charts, stress testing, and scenario-based Monte Carlo"""
    st.subheader("🚨 Advanced Risk Analysis")
    
    # Create tabs for different risk analysis types
    risk_tab1, risk_tab2, risk_tab3, risk_tab4 = st.tabs([
        "🎯 Risk Metrics", "🌪️ Tornado Chart", "💥 Stress Testing", "📊 Scenario Monte Carlo"
    ])
    
    with risk_tab1:
        st.write("**Risk Metrics from Monte Carlo Simulation**")
        if 'mc_results' in st.session_state:
            mc_results = st.session_state.mc_results
            risk_metrics = model.calculate_risk_metrics(mc_results)
            
            if risk_metrics:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("VaR (5%)", f"${risk_metrics.get('VaR_5%', 0):,.0f}")
                    st.metric("Expected Shortfall (5%)", f"${risk_metrics.get('Expected_Shortfall_5%', 0):,.0f}")
                    st.metric("Sharpe Ratio", f"{risk_metrics.get('Sharpe_Ratio', 0):.2f}")
                
                with col2:
                    st.metric("VaR (1%)", f"${risk_metrics.get('VaR_1%', 0):,.0f}")
                    st.metric("Expected Shortfall (1%)", f"${risk_metrics.get('Expected_Shortfall_1%', 0):,.0f}")
                    st.metric("Sortino Ratio", f"{risk_metrics.get('Sortino_Ratio', 0):.2f}")
                
                with col3:
                    st.metric("Upside Potential (5%)", f"${risk_metrics.get('Upside_Potential_5%', 0):,.0f}")
                    st.metric("Best Case Avg (5%)", f"${risk_metrics.get('Best_Case_Avg_5%', 0):,.0f}")
                    st.metric("Upside Potential (1%)", f"${risk_metrics.get('Upside_Potential_1%', 0):,.0f}")
                
                st.write("**Risk Metrics Explanation:**")
                st.write("- **VaR (Value at Risk)**: Maximum expected loss at given confidence level")
                st.write("- **Expected Shortfall**: Average loss in worst-case scenarios")
                st.write("- **Upside Potential**: Maximum expected gain at given confidence level")
                st.write("- **Sharpe Ratio**: Risk-adjusted return (higher is better)")
                st.write("- **Sortino Ratio**: Risk-adjusted return considering only downside risk")
        else:
            st.warning("Please run Monte Carlo simulation first to see risk metrics.")
    
    with risk_tab2:
        st.write("**Tornado Chart - Multi-Variable Sensitivity Analysis**")
        st.write("Shows which variables have the biggest impact on your final cash flow:")
        
        if st.button("Generate Tornado Chart", type="primary"):
            with st.spinner("Generating tornado chart..."):
                tornado_data = model.create_tornado_chart_data(assumptions)
                
                # Create tornado chart
                fig_tornado = go.Figure()
                
                # Add bars for each variable
                for _, row in tornado_data.iterrows():
                    variable = row['Variable']
                    low_impact = row['Low_Impact_%']
                    high_impact = row['High_Impact_%']
                    
                    # Add low scenario bar
                    fig_tornado.add_trace(go.Bar(
                        y=[variable],
                        x=[low_impact],
                        name='Low Scenario',
                        orientation='h',
                        marker_color='red',
                        showlegend=False
                    ))
                    
                    # Add high scenario bar
                    fig_tornado.add_trace(go.Bar(
                        y=[variable],
                        x=[high_impact],
                        name='High Scenario',
                        orientation='h',
                        marker_color='green',
                        showlegend=False
                    ))
                
                fig_tornado.update_layout(
                    title="Tornado Chart - Impact on Final Cash Flow",
                    xaxis_title="Impact on Final Cash Flow (%)",
                    yaxis_title="Variables",
                    barmode='overlay',
                    height=400
                )
                
                st.plotly_chart(fig_tornado, use_container_width=True)
                
                # Display tornado data table
                st.write("**Detailed Impact Analysis:**")
                display_tornado = tornado_data.copy()
                for col in ['Low_Impact_%', 'High_Impact_%', 'Max_Impact_%']:
                    display_tornado[col] = display_tornado[col].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display_tornado, use_container_width=True)
    
    with risk_tab3:
        st.write("**Stress Testing - Extreme Scenario Analysis**")
        st.write("Tests your business model against extreme market conditions:")
        
        if st.button("Run Stress Tests", type="primary"):
            with st.spinner("Running stress tests..."):
                stress_results = model.run_stress_test(assumptions)
                
                # Display stress test results
                stress_data = []
                for scenario, results in stress_results.items():
                    stress_data.append({
                        'Scenario': scenario.replace('_', ' '),
                        'Final Cash Flow': results['final_cash_flow'],
                        'Final AUA': results['final_aua'],
                        'Breakeven Month': results['breakeven_month'],
                        'Year 1 Revenue': results['year1_revenue'],
                        'Survives': 'Yes' if results['survival_probability'] == 1 else 'No'
                    })
                
                stress_df = pd.DataFrame(stress_data)
                
                # Format display
                display_stress = stress_df.copy()
                for col in ['Final Cash Flow', 'Final AUA', 'Year 1 Revenue']:
                    display_stress[col] = display_stress[col].apply(lambda x: f"${x:,.0f}")
                
                st.dataframe(display_stress, use_container_width=True)
                
                # Create stress test visualization
                fig_stress = go.Figure()
                
                scenarios = stress_df['Scenario']
                # Ensure cash_flows is numeric
                if pd.api.types.is_numeric_dtype(stress_df['Final Cash Flow']):
                    cash_flows = stress_df['Final Cash Flow'].astype(float)
                else:
                    cash_flows = stress_df['Final Cash Flow'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
                
                colors = ['red' if cf < 0 else 'green' for cf in cash_flows]
                
                fig_stress.add_trace(go.Bar(
                    x=scenarios,
                    y=cash_flows,
                    marker_color=colors,
                    text=[f"${cf:,.0f}" for cf in cash_flows],
                    textposition='auto'
                ))
                
                fig_stress.add_hline(y=0, line_dash="dash", line_color="black", 
                                   annotation_text="Breakeven Line")
                
                fig_stress.update_layout(
                    title="Stress Test Results - Final Cash Flow",
                    xaxis_title="Stress Scenario",
                    yaxis_title="Final Cash Flow ($)",
                    height=400
                )
                
                st.plotly_chart(fig_stress, use_container_width=True)
    
    with risk_tab4:
        st.write("**Scenario-Based Monte Carlo Simulation**")
        st.write("Monte Carlo with different probability distributions for different market conditions:")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_scenario_sims = st.number_input(
                "Number of Scenario Simulations", 
                min_value=100, max_value=5000, value=1000, step=100
            )
            
            if st.button("Run Scenario Monte Carlo", type="primary"):
                with st.spinner("Running scenario-based Monte Carlo..."):
                    scenario_results = model.run_scenario_based_monte_carlo(assumptions, n_scenario_sims)
                    st.session_state.scenario_results = scenario_results
        
        with col2:
            if 'scenario_results' in st.session_state:
                scenario_results = st.session_state.scenario_results
                
                # Display scenario breakdown
                st.write("**Market Scenario Distribution:**")
                scenario_breakdown = scenario_results['scenario_breakdown']
                total_sims = sum(scenario_breakdown.values())
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    bull_pct = (scenario_breakdown['Bull_Market'] / total_sims) * 100
                    st.metric("Bull Market", f"{bull_pct:.1f}%")
                with col_b:
                    normal_pct = (scenario_breakdown['Normal_Market'] / total_sims) * 100
                    st.metric("Normal Market", f"{normal_pct:.1f}%")
                with col_c:
                    bear_pct = (scenario_breakdown['Bear_Market'] / total_sims) * 100
                    st.metric("Bear Market", f"{bear_pct:.1f}%")
                
                # Calculate and display scenario-based summary stats
                if scenario_results.get('final_cash_flow'):
                    scenario_summary = model.calculate_monte_carlo_summary_stats(scenario_results)
                    st.write("**Scenario-Based Summary Statistics:**")
                    st.dataframe(scenario_summary, use_container_width=True)

def create_scenario_analysis(assumptions):
    """Create scenario analysis with different assumptions"""
    st.subheader("🔄 Scenario Analysis")
    # Create scenario inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Base Case**")
        base_multiplier = 1.0
    with col2:
        st.markdown("**Downside Case**")
        downside_multiplier = st.slider(
            "Downside Multiplier", 
            min_value=0.5, max_value=1.0, value=0.8, step=0.05
        )
    with col3:
        st.markdown("**Upside Case**")
        upside_multiplier = st.slider(
            "Upside Multiplier", 
            min_value=1.0, max_value=2.0, value=1.3, step=0.05
        )
    # Run scenarios
    scenarios = {}
    # Always use a new model with the current projection period
    projection_months = assumptions.get('projection_months', 60)
    # Base case
    model_base = FinancialModel(months=projection_months)
    scenarios['Base Case'] = model_base.calculate_monthly_projections(assumptions)
    # Downside case
    downside_assumptions = assumptions.copy()
    downside_assumptions['aua_growth_rate'] *= downside_multiplier
    downside_assumptions['monthly_client_growth_rate'] *= downside_multiplier
    downside_assumptions['aua_fee_bps'] *= downside_multiplier
    if 'loan_duration_days' not in downside_assumptions:
        downside_assumptions['loan_duration_days'] = 4
    model_downside = FinancialModel(months=projection_months)
    scenarios['Downside'] = model_downside.calculate_monthly_projections(downside_assumptions)
    # Upside case
    upside_assumptions = assumptions.copy()
    upside_assumptions['aua_growth_rate'] *= upside_multiplier
    upside_assumptions['monthly_client_growth_rate'] *= upside_multiplier
    upside_assumptions['aua_fee_bps'] *= upside_multiplier
    if 'loan_duration_days' not in upside_assumptions:
        upside_assumptions['loan_duration_days'] = 4
    model_upside = FinancialModel(months=projection_months)
    scenarios['Upside'] = model_upside.calculate_monthly_projections(upside_assumptions)
    
    # Display scenario comparison
    scenario_comparison = []
    for name, results in scenarios.items():
        scenario_comparison.append({
            'Scenario': name,
            'Breakeven Month': results['breakeven_month'].iloc[-1],
            'Year 1 Revenue': results['total_revenue'].iloc[:12].sum(),
            'Year 2 Revenue': results['total_revenue'].iloc[12:24].sum(),
            'Final AUA': results['total_aua'].iloc[-1],
            'Final Cash Flow': results['cumulative_cash_flow'].iloc[-1]
        })
    
    comparison_df = pd.DataFrame(scenario_comparison)
    
    # Format the comparison table
    display_comparison = comparison_df.copy()
    for col in ['Year 1 Revenue', 'Year 2 Revenue', 'Final AUA', 'Final Cash Flow']:
        display_comparison[col] = display_comparison[col].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(display_comparison, use_container_width=True)
    
    # Scenario charts
    fig_scenarios = go.Figure()
    colors = ['blue', 'red', 'green']
    
    for (name, results), color in zip(scenarios.items(), colors):
        fig_scenarios.add_trace(go.Scatter(
            x=results.index,
            y=results['cumulative_cash_flow'],
            mode='lines',
            name=name,
            line=dict(color=color, width=2)
        ))
    
    fig_scenarios.add_hline(y=0, line_dash="dash", line_color="black", 
                           annotation_text="Breakeven Line")
    fig_scenarios.update_layout(
        title="Scenario Comparison - Cash Flow",
        xaxis_title="Month",
        yaxis_title="Cumulative Cash Flow ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_scenarios, use_container_width=True)
    
    return scenarios

def create_sensitivity_analysis(assumptions):
    """Create sensitivity analysis"""
    st.subheader("📊 Sensitivity Analysis")
    
    # Select variable for sensitivity analysis
    sensitivity_variable = st.selectbox(
        "Select Variable for Sensitivity Analysis",
        ['aua_fee_bps', 'saas_monthly_fee', 'transaction_fee', 'monthly_client_growth_rate']
    )
    
    # Define ranges for each variable
    ranges = {
        'aua_fee_bps': (0, 100),
        'saas_monthly_fee': (0, 2000),
        'transaction_fee': (0, 50),
        'monthly_client_growth_rate': (0, 0.2)
    }
    
    if sensitivity_variable in ranges:
        min_val, max_val = ranges[sensitivity_variable]
        
        sensitivity_df = model.create_sensitivity_table(
            assumptions, sensitivity_variable, min_val, max_val, 20
        )
        
        # Create sensitivity chart
        fig_sensitivity = go.Figure()
        
        fig_sensitivity.add_trace(go.Scatter(
            x=sensitivity_df[sensitivity_variable],
            y=sensitivity_df['breakeven_month'],
            mode='lines+markers',
            name='Breakeven Month',
            line=dict(color='blue', width=2)
        ))
        
        fig_sensitivity.update_layout(
            title=f"Sensitivity Analysis: {sensitivity_variable.replace('_', ' ').title()}",
            xaxis_title=sensitivity_variable.replace('_', ' ').title(),
            yaxis_title="Breakeven Month",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        # Display sensitivity table
        st.dataframe(sensitivity_df, use_container_width=True)

def main():
    """Main application function"""
    
    # Create sidebar inputs
    assumptions = create_sidebar_inputs()
    
    # Initialize model with selected period
    model = FinancialModel(months=assumptions['projection_months'])
    
    # Calculate projections
    results = model.calculate_monthly_projections(assumptions)
    
    # Display dashboard
    display_dashboard(results, assumptions)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Charts", "📊 Monthly Data", "🔄 Scenarios", "🎲 Monte Carlo", "📊 Sensitivity", "🚨 Advanced Risk"
    ])
    
    with tab1:
        display_charts(results)
    
    with tab2:
        display_monthly_table(results)
    
    with tab3:
        scenarios = create_scenario_analysis(assumptions)
    
    with tab4:
        run_monte_carlo_simulation(assumptions)
    
    with tab5:
        create_sensitivity_analysis(assumptions)
    
    with tab6:
        create_advanced_risk_analysis(assumptions)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>💰 Fintech Financial Model | Built with Streamlit, Pandas, and Plotly</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 