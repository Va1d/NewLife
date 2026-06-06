from lb import drank

codes = drank()

print(f"Stock #30: {codes[30]}")
print("\nTop 10 bot-heavy stocks:")
print("Rank | Index | Symbol")
print("-----|-------|-------")

bot_indices = [30, 34, 16, 26, 17, 28, 20, 25, 14, 3]
for rank, idx in enumerate(bot_indices, 1):
    print(f" {rank:2d}  |  {idx:2d}   | {codes[idx]}")
