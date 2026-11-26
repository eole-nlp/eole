Fix `total_cents` in `cart.py`. The function takes a list of item prices in
integer cents and a discount in whole percent (0–100). Apply the discount once
to the subtotal and round the amount charged to the nearest cent, with an
exact half cent rounded up. Keep the existing interface and use integer
arithmetic. Only edit `cart.py`. Inspect the file first and run the tests after
editing.

Once the tests pass, give a short final answer and stop. Do not run any more commands.
