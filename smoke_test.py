import sys
sys.path.insert(0, 'src')

from simulation.engine import HealthRiskLabEngine

engine = HealthRiskLabEngine(start_year=2020, end_year=2021, seed=42)
for _ in range(3):
    r = engine.run_quarter()
    q = r['quarter']
    sc = r['scenario']['name']
    p_ret = r['player']['portfolio_return']
    ai_ret = r['ai']['portfolio_return']
    print(f"  {q}: {sc} | Player: {p_ret:.2%} | AI: {ai_ret:.2%}")

final = engine.get_final_results()
print(f"\nSimulation: Player={final['player_final_score']} | AI={final['ai_final_score']} | Portfolio=${final['final_portfolio_value_m']}M")
print("Simulation engine: OK")

# Test IBNR
from financial.insurance.actuarial import IBNRCalculator
calc = IBNRCalculator()
tri = IBNRCalculator.generate_sample_triangle(n_years=5, seed=42)
cl = calc.chain_ladder(tri)
print(f"Chain Ladder IBNR: ${cl['total_ibnr']:,.0f} - OK")

# Test pharma rNPV
from financial.pharma.rnpv_calculator import RNPVCalculator, PhaseSuccessModel
phase_model = PhaseSuccessModel()
pos = phase_model.compute_adjusted_probability("Oncology", "Phase III → NDA")
print(f"Phase success prob: {pos['adjusted_probability']:.1%} - OK")

calc2 = RNPVCalculator(n_simulations=500)
rnpv = calc2.calculate(2.0, 0.8, 3.0, 11, 0.493)
print(f"rNPV: ${rnpv['rnpv_m']:.0f}M - OK")

print("\n=== ALL MODULE SMOKE TESTS PASSED ===")
