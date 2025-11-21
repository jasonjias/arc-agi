import numpy as np

def draw_green_spiral_debug(grid):
    h, w = grid.shape
    result = np.zeros((h, w), dtype=int)

    top, bottom = 0, h - 1
    left, right = 0, w - 1

    iteration = 0
    while top <= bottom and left <= right:
        iteration += 1
        print(f"\nIteration {iteration}: top={top}, bottom={bottom}, left={left}, right={right}")

        # Draw top row (left to right)
        print(f"  Drawing top row {top}, cols {left} to {right}")
        for c in range(left, right + 1):
            result[top, c] = 3
        top += 1

        # Draw right column (top to bottom)
        print(f"  Drawing right col {right}, rows {top} to {bottom}")
        for r in range(top, bottom + 1):
            result[r, right] = 3
        right -= 1

        # Draw bottom row (right to left) if there's still a row left
        if top <= bottom:
            print(f"  Drawing bottom row {bottom}, cols {right} to {left}")
            for c in range(right, left - 1, -1):
                result[bottom, c] = 3
            bottom -= 1

        # Draw left column (bottom to top) if there's still a column left
        if left <= right:
            print(f"  Drawing left col {left}, rows {bottom} to {top}")
            for r in range(bottom, top - 1, -1):
                result[r, left] = 3
            left += 1

        print(f"  After iteration:")
        print(result)

    return result

test_input = np.zeros((6, 6), dtype=int)
result = draw_green_spiral_debug(test_input)

print("\n\nFinal result:")
print(result)
