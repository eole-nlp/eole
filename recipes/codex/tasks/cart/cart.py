def total_cents(prices, discount_percent):
    """Return the discounted amount charged in cents."""
    return sum(price * (100 - discount_percent) // 100 for price in prices)
