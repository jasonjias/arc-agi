import numpy as np

# Shape 2 (color 2, rotated 90 to be 2x1)
shape = np.array([[2], [2]])
sh, sw = shape.shape
print(f"Shape size: {sh}x{sw}")
print(shape)

holes_row = 5
start_col = 3

print(f"\nPlacing at column {start_col}")
print(f"holes_row = {holes_row}")
print(f"Shape height (sh) = {sh}")
print(f"Shape width (sw) = {sw}")

print(f"\nFor each cell in shape:")
for sr in range(sh):
    for sc in range(sw):
        if shape[sr, sc] != 0:
            target_row = holes_row - sh + 1 + sr
            target_col = start_col + sc
            print(f"  shape[{sr},{sc}] = {shape[sr, sc]} -> out[{target_row},{target_col}]")

print(f"\nExpected: rows 4-5, column 3")
print("Got: row 5, columns 3-4")
print("\nWait, the debug output showed 'columns 3-4' but we placed at start_col=3 with sw=1...")
print("Let me check the actual output again")

# Actually run inline to see
inp = np.array([[0, 0, 0, 0, 0, 0],
                 [0, 2, 2, 0, 0, 0],
                 [0, 0, 0, 0, 3, 3],
                 [0, 0, 0, 0, 3, 3],
                 [0, 0, 0, 0, 0, 0],
                 [1, 0, 1, 0, 0, 1],
                 [1, 1, 1, 1, 1, 1]])

out = np.array([[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 0, 1, 2, 2, 1],
                 [1, 1, 1, 1, 1, 1]])

print(f"\nActual output row 5: {out[5, :]}")
print(f"  Placed color 2 at columns 3-4")
print(f"  But sw=1, so it should only be at column 3!")
print(f"  This means the shape placed was 1x2 (horizontal), not 2x1 (vertical)")
