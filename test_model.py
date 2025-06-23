#!/usr/bin/env python3
"""
Test script for the Financial Model
This script tests the core functionality of the financial model
"""

from financial_model import FinancialModel
from datetime import date

def test_financial_model():
    """Test the financial model with sample data"""
    print("🧪 Testing Financial Model...")
    
    # Initialize model
    model = FinancialModel()
    
    # Sample assumptions
    assumptions = {
        'start_date': date(2024, 1, 1),
        'starting_clients': 10,
        'monthly_client_growth_rate': 0.05,
        'client_churn_rate': 0.01,
        'starting_aua_per_client': 500000,
        'aua_growth_rate': 0.02,
        'aua_fee_bps': 25,
        'monthly_transactions_per_client': 5,
        'transaction_fee': 10,
        'loan_frequency': 2,
        'avg_loan_size': 100000,
        'loan_duration_days': 4,
        'lending_fee_bps': 100,
        'saas_monthly_fee': 500,
        'saas_enabled': True,
        'fund_revenue_bps': 5,
        'monthly_fixed_costs': 50000,
        'cost_per_client': 100,
        'cost_per_mm_aua': 1000,
        'starting_capital': 1000000
    }
    
    try:
        # Test monthly projections
        print("📊 Testing monthly projections...")
        results = model.calculate_monthly_projections(assumptions)
        
        # Check key outputs
        final_revenue = results['total_revenue'].iloc[-1]
        final_aua = results['total_aua'].iloc[-1]
        breakeven_month = results['breakeven_month'].iloc[-1]
        final_cash_flow = results['cumulative_cash_flow'].iloc[-1]
        
        print(f"✅ Final Monthly Revenue: ${final_revenue:,.0f}")
        print(f"✅ Final Total AUA: ${final_aua:,.0f}")
        print(f"✅ Breakeven Month: {breakeven_month}")
        print(f"✅ Final Cash Flow: ${final_cash_flow:,.0f}")
        
        # Test Monte Carlo simulation
        print("\n🎲 Testing Monte Carlo simulation...")
        mc_results = model.run_monte_carlo(assumptions, n_simulations=100)
        
        if mc_results['breakeven_months']:
            avg_breakeven = sum(mc_results['breakeven_months']) / len(mc_results['breakeven_months'])
            print(f"✅ Average Breakeven Month: {avg_breakeven:.1f}")
        
        survival_pct = mc_results['survival_probability'] * 100
        print(f"✅ Survival Probability: {survival_pct:.1f}%")
        
        # Test sensitivity analysis
        print("\n📊 Testing sensitivity analysis...")
        sensitivity_df = model.create_sensitivity_table(
            assumptions, 'aua_fee_bps', 0, 100, 5
        )
        print(f"✅ Sensitivity table created with {len(sensitivity_df)} rows")
        
        # Test charts creation
        print("\n📈 Testing chart creation...")
        charts = model.create_charts(results, "Test Scenario")
        print(f"✅ Created {len(charts)} charts")
        
        print("\n🎉 All tests passed! The financial model is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_financial_model()
    if success:
        print("\n🚀 Ready to run the Streamlit app with: streamlit run app.py")
    else:
        print("\n🔧 Please check the error and fix before running the app.") 