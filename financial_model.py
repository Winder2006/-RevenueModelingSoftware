import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

class FinancialModel:
    def __init__(self, months: int = 60):
        self.months = months  # projection period in months
        self.scenarios = {}
        
    def calculate_monthly_projections(self, assumptions: Dict) -> pd.DataFrame:
        """Calculate monthly financial projections based on assumptions"""
        # Use the projection period from assumptions if provided
        months = assumptions.get('projection_months', self.months)
        dates = pd.date_range(start=assumptions['start_date'], periods=months, freq='ME')
        results = pd.DataFrame(index=dates)
        # If months == 1, just use the current values (no forecasting)
        if months == 1:
            results['total_clients'] = [assumptions['starting_clients']]
            results['aua_per_client'] = [assumptions['starting_aua_per_client']]
            results['total_aua'] = [assumptions['starting_clients'] * assumptions['starting_aua_per_client']]
            results['aua_revenue'] = [results['total_aua'][0] * assumptions['aua_fee_bps'] / 10000]
            results['transaction_revenue'] = [assumptions['starting_clients'] * assumptions['monthly_transactions_per_client'] * assumptions['transaction_fee']]
            results['lending_revenue'] = [assumptions['starting_clients'] * (assumptions['loan_frequency']/12) * assumptions['avg_loan_size'] * assumptions['lending_fee_bps'] / 10000]
            results['saas_revenue'] = [assumptions['starting_clients'] * assumptions['saas_monthly_fee'] if assumptions['saas_enabled'] else 0]
            results['fund_revenue'] = [results['total_aua'][0] * assumptions['fund_revenue_bps'] / 10000]
            results['total_revenue'] = results['aua_revenue'] + results['transaction_revenue'] + results['lending_revenue'] + results['saas_revenue'] + results['fund_revenue']
            results['fixed_costs'] = [assumptions['monthly_fixed_costs']]
            results['variable_costs'] = [assumptions['starting_clients'] * assumptions['cost_per_client'] + results['total_aua'][0] * assumptions['cost_per_mm_aua'] / 1000000]
            results['total_costs'] = results['fixed_costs'] + results['variable_costs']
            results['net_income'] = results['total_revenue'] - results['total_costs']
            results['cumulative_cash_flow'] = results['net_income'] - assumptions['starting_capital']
            results['runway_months'] = [0]
            results['breakeven_month'] = [-1]
            results['breakeven_aua'] = [results['total_aua'][0]]
            return results
        # Otherwise, use the normal forecasting logic
        self.months = months
        
        # Client growth calculations
        results['total_clients'] = self._calculate_client_growth(
            assumptions['starting_clients'],
            assumptions['monthly_client_growth_rate'],
            assumptions['client_churn_rate'],
            dates
        )
        
        # AUA calculations
        results['aua_per_client'] = self._calculate_aua_per_client(
            assumptions['starting_aua_per_client'],
            assumptions['aua_growth_rate'],
            dates
        )
        results['total_aua'] = results['total_clients'] * results['aua_per_client']
        
        # Revenue calculations
        results['aua_revenue'] = (results['total_aua'] * assumptions['aua_fee_bps'] / 10000)
        results['transaction_revenue'] = self._calculate_transaction_revenue(
            results['total_clients'],
            assumptions['monthly_transactions_per_client'],
            assumptions['transaction_fee']
        )
        # Use loan_duration_days, default to 4 if not present
        loan_duration_days = assumptions.get('loan_duration_days', 4)
        results['lending_revenue'] = self._calculate_lending_revenue(
            results['total_clients'],
            assumptions['loan_frequency'],
            assumptions['avg_loan_size'],
            loan_duration_days,
            assumptions['lending_fee_bps']
        )
        results['saas_revenue'] = self._calculate_saas_revenue(
            results['total_clients'],
            assumptions['saas_monthly_fee'],
            assumptions['saas_enabled']
        )
        results['fund_revenue'] = self._calculate_fund_revenue(
            results['total_aua'],
            assumptions['fund_revenue_bps']
        )
        
        # Total revenue
        results['total_revenue'] = (results['aua_revenue'] + results['transaction_revenue'] + 
                                  results['lending_revenue'] + results['saas_revenue'] + 
                                  results['fund_revenue'])
        
        # Cost calculations
        results['fixed_costs'] = assumptions['monthly_fixed_costs']
        results['variable_costs'] = self._calculate_variable_costs(
            results['total_clients'],
            results['total_aua'],
            assumptions
        )
        results['total_costs'] = results['fixed_costs'] + results['variable_costs']
        
        # Profitability
        results['net_income'] = results['total_revenue'] - results['total_costs']
        results['cumulative_cash_flow'] = results['net_income'].cumsum() - assumptions['starting_capital']
        
        # Key metrics
        results['runway_months'] = self._calculate_runway(results['cumulative_cash_flow'])
        results['breakeven_month'] = self._find_breakeven_month(results['cumulative_cash_flow'])
        results['breakeven_aua'] = self._calculate_breakeven_aua(results, assumptions)
        
        return results
    
    def _calculate_client_growth(self, starting_clients: float, growth_rate: float, churn_rate: float, dates: pd.DatetimeIndex) -> pd.Series:
        """Calculate client growth with churn"""
        clients = [starting_clients]
        for i in range(1, self.months):
            new_clients = clients[i-1] * (1 + growth_rate - churn_rate)
            clients.append(max(0, new_clients))
        return pd.Series(clients, index=dates)
    
    def _calculate_aua_per_client(self, starting_aua: float, growth_rate: float, dates: pd.DatetimeIndex) -> pd.Series:
        """Calculate AUA per client growth"""
        aua = [starting_aua]
        for i in range(1, self.months):
            new_aua = aua[i-1] * (1 + growth_rate)
            aua.append(new_aua)
        return pd.Series(aua, index=dates)
    
    def _calculate_transaction_revenue(self, clients: pd.Series, transactions_per_client: float, fee: float) -> pd.Series:
        """Calculate transaction-based revenue"""
        return clients * transactions_per_client * fee
    
    def _calculate_lending_revenue(self, clients: pd.Series, loan_frequency: float, avg_loan_size: float, 
                                 loan_duration_days: float, fee_bps: float) -> pd.Series:
        """Calculate lending revenue (duration in days)"""
        # Convert loan duration from days to years for annualization
        loan_duration_years = loan_duration_days / 365
        monthly_loans = clients * loan_frequency / 12
        loan_value = monthly_loans * avg_loan_size
        # Revenue is based on the value of loans and fee bps
        return loan_value * fee_bps / 10000
    
    def _calculate_saas_revenue(self, clients: pd.Series, monthly_fee: float, enabled: bool) -> pd.Series:
        """Calculate SaaS revenue"""
        if enabled:
            return clients * monthly_fee
        return pd.Series([0] * len(clients), index=clients.index)
    
    def _calculate_fund_revenue(self, total_aua: pd.Series, fee_bps: float) -> pd.Series:
        """Calculate fund-side revenue"""
        return total_aua * fee_bps / 10000
    
    def _calculate_variable_costs(self, clients: pd.Series, total_aua: pd.Series, assumptions: Dict) -> pd.Series:
        """Calculate variable costs"""
        client_costs = clients * assumptions['cost_per_client']
        aua_costs = total_aua * assumptions['cost_per_mm_aua'] / 1000000
        return client_costs + aua_costs
    
    def _calculate_runway(self, cumulative_cash_flow: pd.Series) -> pd.Series:
        """Calculate runway in months"""
        runway = []
        for i, cash_flow in enumerate(cumulative_cash_flow):
            if cash_flow >= 0:
                runway.append(0)
            else:
                # Find how many months until cash flow becomes positive
                future_cash_flows = cumulative_cash_flow[i:]
                positive_months = (future_cash_flows >= 0).sum()
                if positive_months > 0:
                    runway.append(len(future_cash_flows) - positive_months)
                else:
                    runway.append(len(future_cash_flows))
        return pd.Series(runway, index=cumulative_cash_flow.index)
    
    def _find_breakeven_month(self, cumulative_cash_flow: pd.Series) -> int:
        """Find the month when breakeven occurs"""
        positive_months = (cumulative_cash_flow >= 0)
        if positive_months.any():
            # Find the first month where cash flow becomes positive
            breakeven_idx = positive_months.idxmax()
            # Return the month number (1-based)
            return cumulative_cash_flow.index.get_loc(breakeven_idx) + 1
        return -1  # Never breakeven
    
    def _calculate_breakeven_aua(self, results: pd.DataFrame, assumptions: Dict) -> float:
        """Calculate AUA needed to breakeven"""
        breakeven_month = results['breakeven_month'].iloc[-1]
        if breakeven_month > 0 and breakeven_month <= len(results):
            return results['total_aua'].iloc[breakeven_month - 1]
        return results['total_aua'].iloc[-1]  # Return final AUA if no breakeven
    
    def run_monte_carlo(self, assumptions: Dict, n_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation"""
        results = {
            'breakeven_months': [],
            'year1_revenue': [],
            'year2_revenue': [],
            'year3_revenue': [],
            'survival_probability': 0,
            'aua_variability': []
        }
        
        for _ in range(n_simulations):
            # Add random variation to key assumptions
            sim_assumptions = assumptions.copy()
            # Add noise to key variables
            sim_assumptions['aua_growth_rate'] += np.random.normal(0, 0.001)  # 0.1% std dev
            sim_assumptions['monthly_client_growth_rate'] += np.random.normal(0, 0.002)  # 0.2% std dev
            sim_assumptions['client_churn_rate'] += np.random.normal(0, 0.001)  # 0.1% std dev
            sim_assumptions['aua_fee_bps'] += np.random.normal(0, 0.5)  # 0.5 bps std dev
            # Ensure reasonable bounds
            sim_assumptions['aua_growth_rate'] = max(0, min(0.05, sim_assumptions['aua_growth_rate']))
            sim_assumptions['monthly_client_growth_rate'] = max(0, min(0.1, sim_assumptions['monthly_client_growth_rate']))
            sim_assumptions['client_churn_rate'] = max(0, min(0.05, sim_assumptions['client_churn_rate']))
            sim_assumptions['aua_fee_bps'] = max(0, sim_assumptions['aua_fee_bps'])
            # Run simulation
            sim_results = self.calculate_monthly_projections(sim_assumptions)
            # Collect results
            breakeven_month = sim_results['breakeven_month'].iloc[-1]
            if breakeven_month > 0:
                results['breakeven_months'].append(breakeven_month)
            # Only sum available months for each year
            n_months = len(sim_results['total_revenue'])
            results['year1_revenue'].append(sim_results['total_revenue'].iloc[:min(12, n_months)].sum())
            results['year2_revenue'].append(sim_results['total_revenue'].iloc[12:min(24, n_months)].sum() if n_months > 12 else 0)
            results['year3_revenue'].append(sim_results['total_revenue'].iloc[24:min(36, n_months)].sum() if n_months > 24 else 0)
            # Check survival (positive cash flow within available months or 36 if possible)
            check_month = min(35, n_months - 1)
            if sim_results['cumulative_cash_flow'].iloc[check_month] >= 0:
                results['survival_probability'] += 1
            results['aua_variability'].append(sim_results['total_aua'].iloc[-1])
        results['survival_probability'] /= n_simulations
        return results
    
    def create_sensitivity_table(self, base_assumptions: Dict, variable: str, 
                               min_val: float, max_val: float, steps: int = 10) -> pd.DataFrame:
        """Create sensitivity analysis table"""
        values = np.linspace(min_val, max_val, steps)
        results = []
        
        for val in values:
            assumptions = base_assumptions.copy()
            assumptions[variable] = val
            projections = self.calculate_monthly_projections(assumptions)
            
            results.append({
                variable: val,
                'breakeven_month': projections['breakeven_month'].iloc[-1],
                'year1_revenue': projections['total_revenue'].iloc[:12].sum(),
                'year2_revenue': projections['total_revenue'].iloc[12:24].sum(),
                'final_aua': projections['total_aua'].iloc[-1],
                'cumulative_cash_flow': projections['cumulative_cash_flow'].iloc[-1]
            })
        
        return pd.DataFrame(results)
    
    def create_charts(self, results: pd.DataFrame, scenario_name: str = "Base Case") -> Dict:
        """Create interactive charts"""
        charts = {}
        
        # Cash burn curve
        fig_cash_burn = go.Figure()
        fig_cash_burn.add_trace(go.Scatter(
            x=results.index,
            y=results['cumulative_cash_flow'],
            mode='lines+markers',
            name='Cumulative Cash Flow',
            line=dict(color='blue', width=2)
        ))
        fig_cash_burn.add_hline(y=0, line_dash="dash", line_color="red", 
                               annotation_text="Breakeven Line")
        fig_cash_burn.update_layout(
            title=f'Cash Burn Curve - {scenario_name}',
            xaxis_title='Month',
            yaxis_title='Cumulative Cash Flow ($)',
            hovermode='x unified'
        )
        charts['cash_burn'] = fig_cash_burn
        
        # Revenue vs Costs
        fig_revenue_costs = go.Figure()
        fig_revenue_costs.add_trace(go.Scatter(
            x=results.index,
            y=results['total_revenue'],
            mode='lines',
            name='Total Revenue',
            line=dict(color='green', width=2)
        ))
        fig_revenue_costs.add_trace(go.Scatter(
            x=results.index,
            y=results['total_costs'],
            mode='lines',
            name='Total Costs',
            line=dict(color='red', width=2)
        ))
        fig_revenue_costs.update_layout(
            title=f'Revenue vs Costs - {scenario_name}',
            xaxis_title='Month',
            yaxis_title='Amount ($)',
            hovermode='x unified'
        )
        charts['revenue_costs'] = fig_revenue_costs
        
        # Revenue mix
        fig_revenue_mix = go.Figure()
        revenue_columns = ['aua_revenue', 'transaction_revenue', 'lending_revenue', 
                          'saas_revenue', 'fund_revenue']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for col, color in zip(revenue_columns, colors):
            if col in results.columns and results[col].sum() > 0:
                fig_revenue_mix.add_trace(go.Scatter(
                    x=results.index,
                    y=results[col],
                    mode='lines',
                    name=col.replace('_', ' ').title(),
                    line=dict(color=color, width=2),
                    stackgroup='one'
                ))
        
        fig_revenue_mix.update_layout(
            title=f'Revenue Mix Over Time - {scenario_name}',
            xaxis_title='Month',
            yaxis_title='Revenue ($)',
            hovermode='x unified'
        )
        charts['revenue_mix'] = fig_revenue_mix
        
        # AUA Growth
        fig_aua_growth = go.Figure()
        fig_aua_growth.add_trace(go.Scatter(
            x=results.index,
            y=results['total_aua'],
            mode='lines+markers',
            name='Total AUA',
            line=dict(color='purple', width=2)
        ))
        fig_aua_growth.update_layout(
            title=f'AUA Growth - {scenario_name}',
            xaxis_title='Month',
            yaxis_title='AUA ($)',
            hovermode='x unified'
        )
        charts['aua_growth'] = fig_aua_growth
        
        return charts 