

def uk_income_tax_2024(gross_income: float) -> float:
    """UK PAYE tax bands 2024/25 (ignores NI for now)."""
    personal_allowance = 12570.0
    if gross_income > 100000:
        reduction = (gross_income - 100000.0) / 2.0
        personal_allowance = max(0.0, personal_allowance - reduction)
    taxable = max(0.0, gross_income - personal_allowance)
    tax = 0.0
    basic_band = 37700.0
    higher_band = 75000.0
    basic_tax = min(taxable, basic_band)
    tax += basic_tax * 0.20
    remaining = taxable - basic_tax
    if remaining > 0:
        higher_tax = min(remaining, higher_band)
        tax += higher_tax * 0.40
        remaining -= higher_tax
    if remaining > 0:
        tax += remaining * 0.45
    return tax
