"""Simple check: Does pixel counting match expected?"""
import sys
sys.path.insert(0, 'src')

from primitives.benchmark_numerosity import PureNumerosityBenchmark

benchmark = PureNumerosityBenchmark()

print("Checking if expected_count == non-zero pixels:\n")

for puzzle in benchmark.puzzles:
    grid = puzzle['grid']
    name = puzzle['name']
    expected = puzzle['expected_total']
    pixel_count = (grid > 0).sum()
    
    match = "✓" if pixel_count == expected else "✗"
    print(f"{match} {name:20s}: Pixels={pixel_count:2d}, Expected={expected:2d}, Match={pixel_count==expected}")

print("\n" + "="*60)
print("INSIGHT:")
pixel_matches = sum((grid > 0).sum() == puzzle['expected_total'] for grid, puzzle in [(p['grid'], p) for p in benchmark.puzzles])
print(f"Pixel count matches expected in {pixel_matches}/{len(benchmark.puzzles)} cases")

if pixel_matches == len(benchmark.puzzles):
    print("\n✓✓ PERFECT! Expected count = non-zero pixels!")
    print("This means we just need to COUNT NON-ZERO PIXELS accurately!")
else:
    print(f"\n⚠ Only {pixel_matches/len(benchmark.puzzles)*100:.0f}% match - some puzzles count objects differently")
